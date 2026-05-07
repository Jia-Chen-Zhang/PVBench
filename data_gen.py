import random
import os
import argparse
import time
import openai
import numpy as np
import sys
# import torch
import subprocess
import json
import shutil

# from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from utils.utils import *
from utils.dataload import load_data
from utils.parser import *
# from utils.cuda_available import cuda_available
from utils.python_executor import PythonExecutor
from eval.evaluate import evaluate
from src.entity_extraction import *
from src.entity_score import extract_entities_and_scores, average_entities_scores, average_entity_scores, extract_entity_scores
from src.entity_summary import find_optimal, find_alter_optimal
from utils.python_executor import PythonExecutor
from utils.self_consistency import aggregate_final_answer


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# initial_system_prompt = "As one of the most distinguished mathematicians, logicians, programmers, and AI scientists, you possess an unparalleled mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry. You are not only intelligent and rational but also prudent and cautious. You are willing to write and execute Python code. Let's approach each problem step by step, take a deep breath, do not save your words, articulating our thoughts in detail, as detailed as possible."
initial_system_prompt = ""
def read_and_print_file(file_path):
    """
    读取指定路径的文件并打印其内容。

    参数:
    - file_path (str): 文件的路径。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
    except Exception as e:
        print(f"读取文件时出错：{e}")


def get_git_commit_id(length=7):
    try:
        # Run the git command to get the latest commit ID
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        # Decode bytes to string
        return commit_id.decode('utf-8')[:length]
    except subprocess.CalledProcessError:
        # Handle errors if the command fails
        return "Unknown"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# define a retry decorator
def retry_with_exponential_backoff(
    func,  # 要重试的函数
    initial_delay: float = 1,  # 初始重试延迟时间，单位为秒
    exponential_base: float = 2,  # 指数退避的基数
    max_delay: float = 8,  # 最大重试延迟时间，防止退避时间过长
    jitter: bool = True,  # 是否添加随机抖动，用于防止雷达效应
    max_retries: int = 20,  # 最大重试次数
    errors: tuple = (openai.error.RateLimitError, openai.error.APIConnectionError, openai.error.APIError, openai.error.ServiceUnavailableError),
    # 可以触发重试机制的异常类元组
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # 定义装饰器内部的包装函数
        # Initialize variables
        num_retries = 0  # 当前重试次数计数
        delay = initial_delay  # 当前延迟时间，从初始延迟开始

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:  # 循环尝试执行函数，直到成功或达到最大重试次数
            try:
                return func(*args, **kwargs)  # 尝试执行函数，成功则直接返回结果
            
            # Retry on specified errors
            except errors as e:  # 捕捉定义的可以触发重试的异常
                # Increment retries
                num_retries += 1  # 重试次数增加
                print("<error>", e, "</error>")  # 打印错误信息

                # Check if max retries has been reached
                if num_retries > max_retries:  # 如果超过最大重试次数
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )  # 抛出异常，中断执行

                # Increment the delay
                delay *= min(exponential_base * (1 + jitter * random.random()), max_delay)  # 计算下次重试的延迟时间，考虑抖动和最大延迟

                # Sleep for the delay
                time.sleep(delay)  # 按计算的延迟时间等待

            # Raise exceptions for any errors not specified
            except Exception as e:  # 捕获未在errors中指定的其他异常
                raise e  # 直接抛出这些异常

    return wrapper  # 返回装饰器函数



# @retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(20))
@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    # 设置数据集名称，默认为"math"
    parser.add_argument("--data_name", default="gsm8k", type=str)
    # 指定使用的模型或路径，默认为"gpt-3.5-turbo"
    parser.add_argument("--model_name_or_path", default="gpt-3.5-turbo", type=str)
    # 设定提示的类型，默认为"dd"
    parser.add_argument("--prompt_type", default="lot", type=str)
    # 定义数据的分割类型，默认为"study"
    parser.add_argument("--split", default="only_debug", type=str)
    # 设置测试样本数量，默认为-1，表示使用全部数据
    parser.add_argument("--num_test_sample", default=-1, type=int) 
    # 设置随机种子，默认为0
    parser.add_argument("--seed", default=0, type=int)
    # 设置处理数据的起始点，默认为0
    parser.add_argument("--start", default=0, type=int)
    # 设置处理数据的终点，默认为-1，通常表示处理到数据集的末尾
    parser.add_argument("--end", default=-1, type=int)
    # 设定模型的温度参数，用于控制生成的多样性，默认为0
    parser.add_argument("--temperature", default=0, type=float)
    # 设置采样次数，默认为1
    parser.add_argument("--n_sampling", default=1, type=int)
    # 设置Top-P采样概率，用于生成文本的随机性控制，默认为0.95
    parser.add_argument("--top_p", default=0.95, type=float)
    # 是否打乱数据，默认为不打乱
    parser.add_argument("--shuffle", action="store_true")
    # 是否使用训练时的提示格式，默认不使用
    parser.add_argument("--use_train_prompt_format", action="store_true")
    # 是否将代码段进行拼接，默认不拼接
    parser.add_argument("--code_concat", action="store_true")
    # 是否开启代码执行警告，默认关闭
    parser.add_argument("--code_exec_warning", action="store_true")
    # 设置最大函数调用次数，默认为4次
    parser.add_argument("--max_func_call", default=4, type=int)
    # 设置代码修正重试的最大次数，默认为4次
    parser.add_argument("--max_code_fix_retries", default=4, type=int)
    # 是否开启详细日志输出，默认关闭
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--scnum", default=5, type=int)
    parser.add_argument("--max_k", default=9, type=int)    
    parser.add_argument("--name_of_dataset", default="gsmak", type=str)     
 



    args = parser.parse_args()


    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    args.max_code_fix_retries = min(args.max_code_fix_retries, int(args.max_func_call / 2))
    if args.prompt_type in ["dd"]:
        args.max_func_call = max(args.max_func_call, 10)
    return args

def extract_json(string):
    try:
        string = string.strip()
        start, end = string.find("{"), string.rfind("}")
        if start != -1 and end != -1:
            string = string[start : end + 1]
        json_data = json.loads(string)
        return json_data
    except Exception as e:
        return str(e)

def extract_request_id(response):
    """
    从 API 响应中提取 request_id（兼容多种格式）
    """
    # 1. 尝试从对象属性获取
    if hasattr(response, 'id'):
        return response.id
    
    # 2. 尝试从字典获取
    if isinstance(response, dict) and 'id' in response:
        return response['id']
    
    # 3. 尝试从响应头获取
    if hasattr(response, 'headers'):
        headers = response.headers
        if isinstance(headers, dict):
            return headers.get('x-request-id') or headers.get('X-Request-ID')
    
    # 4. 尝试从 _headers 获取（OpenAI SDK 内部）
    if hasattr(response, '_headers') and response._headers:
        return response._headers.get('x-request-id')
    
    return None

def gpt_gen_lot(args, stop_tokens, prompts):
    response = completion_with_backoff(
        model=args.model_name_or_path,
        messages=[{"role": "user", "content": prompts}],
        max_tokens=4096,
        # streaming=True,
        # enable_thinking=False,
        temperature=args.temperature,
        # top_p=args.top_p,
        # stop=stop_tokens
    )
    # print("Response:", response)
    # request_id = extract_request_id(response)
    content = response["choices"][0]["message"]["content"]
    # print(f"📋 [Request ID: {request_id}] 生成完成，长度：{len(content)}")
    print("first_outputs:", content)

    return content

def gpt_gen_lot_2(args, stop_tokens, prompts):
    api_base="https://api.deepseek.com/v1"
    api_key="sk-e77c9c411c51435db9afcf97e63cd44f"
    model = "deepseek-v4-flash"
    response = completion_with_backoff(
        model= model,
        messages=[{"role": "user", "content": prompts}],
        max_tokens=8192,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop_tokens,
        enable_thinking=False,
        api_base=api_base,  # 注意：部分库可能要求 base_url
        api_key=api_key
    )
    content = response["choices"][0]["message"]["content"]
    print("second_outputs:", content)

    return content


def gpt_gen_action(args, stop_tokens, prompts):
    response = completion_with_backoff(
        model=args.model_name_or_path,
        messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompts}],
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop_tokens
    )
    content = response["choices"][0]["message"]["content"]
    # print("action_outputs:", content)

    return content

def gpt_gen_infer(args, stop_tokens, prompts):
    response = completion_with_backoff(
        model=args.model_name_or_path,
        messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompts}],
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop_tokens
    )
    content = response["choices"][0]["message"]["content"]
    print("action_outputs:", content)

    return content

def gpt_gen_decompose(args, stop_tokens, prompts):
    response = completion_with_backoff(
        model=args.model_name_or_path,
        messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompts}],
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop_tokens
    )
    content = response["choices"][0]["message"]["content"]
    # print("decompose_outputs:", content)

    return content

def gpt_gen_clean(args, stop_tokens, prompts):
    response = completion_with_backoff(
        model=args.model_name_or_path,
        messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompts}],
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop_tokens
    )
    content = response["choices"][0]["message"]["content"]

    return content


def gpt_gen_entity(args, stop_tokens, prompts):
    entity_outputs = []
    for prompt in tqdm(prompts, desc="Get entity and event hints"):  # 对每个提示进行遍历，并显示进度
            # 使用OpenAI API进行请求
            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompt[1]}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens
            )
            # 将API返回的内容添加到输出列表中
            entity_outputs.append(response["choices"][0]["message"]["content"])

    return entity_outputs


def entity_to_txt(samples, entity_outputs, out_file):
    entity_hint_outputs = []
    remain_entities_hints = []
    for output in entity_outputs:
        entity_hint_outputs.append(extract_entities_and_hints(output))
        remain_entities_hints.append(extract_entities_and_hints(output))

    for entity, example in zip(entity_hint_outputs, samples):
        txt_output_entity = str(entity)
        file_idx = example['idx']
        entity_hint_dir = out_file + '/entity_hint_com'
        os.makedirs(entity_hint_dir, exist_ok=True)
        output_entity_hint_file = entity_hint_dir + f'/{file_idx}.txt'
        json_output = f"idx:{file_idx}\n\nQuestion:{example['question']}\n\n" + txt_output_entity
        with open(output_entity_hint_file, 'w') as f:
            f.write(json_output)
    return entity_hint_outputs



def gpt_gen_entity_score(args, stop_tokens, samples, entity_hint_outputs):
    score_prompts = []
    score_outputs = []

    for sample, entity_hint_output in tqdm(zip(samples, entity_hint_outputs), total=len(samples), desc="Build scoring prompts"):
    # 这里执行你的代码逻辑
        score_prompt = construct_scores_prompt(args, sample, entity_hint_output)#这里使用cr会使具有对应数据集的md文件添加进入prompt中
        score_prompts.append(score_prompt)  

    for prompt in tqdm(score_prompts, desc="Initial scoring"):  # 对每个提示进行遍历，并显示进度
            # 使用OpenAI API进行请求
            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens
            )
            # 将API返回的内容添加到输出列表中
            score_outputs.append(response["choices"][0]["message"]["content"])

    return score_outputs


def score_to_txt(samples, score_outputs, out_file):
    remain_entities = []
    entity_score_outputs = []
    entity_avescore_pair = []
    for output in score_outputs:
        entity_score_output = extract_entities_and_scores(output)
        entity_score_outputs.append(entity_score_output)
        entity_avescore, entities_list = average_entities_scores(entity_score_output)
        entity_avescore_pair.append(entity_avescore)
        remain_entities.append(entities_list)
    #print(entity_avescore_pair)

 
    for entity_score, example  in zip(entity_score_outputs, samples):
        json_output_score = str(entity_score)
        file_idx = example['idx']
        entity_score_dir = out_file + '/score'
        os.makedirs(entity_score_dir, exist_ok=True)
        output_entity_hint_file = entity_score_dir + f'/{file_idx}.txt'
        json_output = f"idx:{file_idx}\n\nQuestion:{example['question']}\n\n" + json_output_score 
        with open(output_entity_hint_file, 'w') as f:
            f.write(json_output)

    return remain_entities, entity_avescore_pair



def gpt_gen_for_alter_summary(args, stop_tokens, samples, entity_hint_outputs, entity_avescore_pair, out_file):
    final_hint = []
    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call
    for sample, ideas, avescore in tqdm(zip(samples, entity_hint_outputs, entity_avescore_pair), total=len(samples), desc="Conduct a cycle summary"):
        file_idx = sample['idx']
        for_dir = out_file + '/for'
        os.makedirs(for_dir, exist_ok=True)   
        output_for_file = for_dir + f'/{file_idx}.txt'
        for_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\n"

        for epoch in range(max_func_call):
            print("=" * 50, "Epoch", epoch)
            print(str(ideas))
            print(avescore)

            if len(ideas.entities) == 1:
                tmp_entity_name = list(ideas.entities.values())[0]
                final_hint.append(tmp_entity_name)
                with open(output_for_file, 'w') as f:
                    f.write(for_output)
                break
            first_entity, second_entity, match = find_alter_optimal(ideas, avescore)
            #解决找不到的问题，一般问题是数量没对上，就是score数量少了,那么就重新评价一次分数
            review_turn = 0
            while not match:
                if review_turn >= 5:
                    break
                score_prompt = construct_scores_prompt(args, sample, ideas)
               
                response = completion_with_backoff(
                    model=args.model_name_or_path,
                    messages=[{"role": "system", "content": initial_system_prompt},
                            {"role": "user", "content": score_prompt}],
                    max_tokens=2048,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=stop_tokens
                )
                # 将API返回的内容添加到输出列表中
                tmp_score_output = response["choices"][0]["message"]["content"]
                tmp_score =extract_entities_and_scores(tmp_score_output)
                tmp_av_score, _ = average_entities_scores(tmp_score)
                avescore = tmp_av_score
                first_entity, second_entity, match = find_alter_optimal(ideas, avescore)
                review_turn += 1
                

            if review_turn >= 5:
                if len(ideas.entities) >= 1:
                    select_entity = list(ideas.entities.values())[0]
                else:
                                        # 创建EntityHints类的实例，名字为"空"
                    empty_entity = EntityHints("Empty")

                    # 向这个实例中添加一个hint
                    empty_entity.add_hint("you have to think by yourself")

                    # 打印这个实例的内容来验证hint是否已正确添加
                    print(empty_entity)

                    select_entity = empty_entity  # 或者你可以选择一个适合你需要的特定空值
                final_hint.append(select_entity)
                break


            if len(ideas.entities) == 1:
                tmp_entity_name = list(ideas.entities.values())[0]
                final_hint.append(tmp_entity_name)
                with open(output_for_file, 'w') as f:
                    f.write(for_output)
                break

            summary_prompt = construct_summary_prompt(args, sample, first_entity, second_entity, ideas)
            # 使用OpenAI API进行请求
            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": summary_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens
            )
            # 将API返回的内容添加到输出列表中
            summary_output = response["choices"][0]["message"]["content"]

            for_output = for_output + str(ideas) + "\n\n"
            ideas.remove_entity(first_entity)
            ideas.remove_entity(second_entity)
            if first_entity in avescore and second_entity in avescore:
                del avescore[first_entity]
                del avescore[second_entity]
            
            ideas, new_entity_name = extract_entitiy_and_hints(ideas, summary_output)

            #print(str(ideas))

            score_prompt = construct_score_prompt(args, sample, ideas, new_entity_name)

            # 使用OpenAI API进行请求
            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": score_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens
            )
            # 将API返回的内容添加到输出列表中
            score_outputs = response["choices"][0]["message"]["content"]

            new_idea_scores_pair = extract_entity_scores(score_outputs, new_entity_name)
            new_idea_averscore = average_entity_scores(new_idea_scores_pair)        
            avescore[new_entity_name] = new_idea_averscore
            #print(avescore)

            if len(ideas.entities) == 1:
                final_hint.append(ideas.entities[new_entity_name])
                with open(output_for_file, 'w') as f:
                    f.write(for_output)
                break
    return final_hint



def gpt_gen_for_summary(args, stop_tokens, samples, entity_hint_outputs, entity_avescore_pair, out_file):
    final_hint = []
    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call
    for sample, ideas, avescore in tqdm(zip(samples, entity_hint_outputs, entity_avescore_pair), total=len(samples), desc="Conduct a cycle summary"):
        file_idx = sample['idx']
        for_dir = out_file + '/for'
        os.makedirs(for_dir, exist_ok=True)   
        output_for_file = for_dir + f'/{file_idx}.txt'
        for_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\n"

        for epoch in range(max_func_call):
            print("=" * 50, "Epoch", epoch)
            print(str(ideas))
            print(avescore)

            if len(ideas.entities) == 1:
                tmp_entity_name = list(ideas.entities.values())[0]
                final_hint.append(tmp_entity_name)
                with open(output_for_file, 'w') as f:
                    f.write(for_output)
                break
            first_entity, second_entity, match = find_optimal(ideas, avescore)
            #解决找不到的问题，一般问题是数量没对上，就是score数量少了,那么就重新评价一次分数
            review_turn = 0
            while not match:
                if review_turn >= 5:
                    break
                score_prompt = construct_scores_prompt(args, sample, ideas)
               
                response = completion_with_backoff(
                    model=args.model_name_or_path,
                    messages=[{"role": "system", "content": initial_system_prompt},
                            {"role": "user", "content": score_prompt}],
                    max_tokens=2048,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=stop_tokens
                )
                # 将API返回的内容添加到输出列表中
                tmp_score_output = response["choices"][0]["message"]["content"]
                tmp_score =extract_entities_and_scores(tmp_score_output)
                tmp_av_score, _ = average_entities_scores(tmp_score)
                avescore = tmp_av_score
                first_entity, second_entity, match = find_optimal(ideas, avescore)
                review_turn += 1
                

            if review_turn >= 5:
                if len(ideas.entities) >= 1:
                    select_entity = list(ideas.entities.values())[0]
                else:
                                        # 创建EntityHints类的实例，名字为"空"
                    empty_entity = EntityHints("Empty")

                    # 向这个实例中添加一个hint
                    empty_entity.add_hint("you have to think by yourself")

                    # 打印这个实例的内容来验证hint是否已正确添加
                    print(empty_entity)

                    select_entity = empty_entity  # 或者你可以选择一个适合你需要的特定空值
                final_hint.append(select_entity)
                break


            if len(ideas.entities) == 1:
                tmp_entity_name = list(ideas.entities.values())[0]
                final_hint.append(tmp_entity_name)
                with open(output_for_file, 'w') as f:
                    f.write(for_output)
                break

            summary_prompt = construct_summary_prompt(args, sample, first_entity, second_entity, ideas)
            # 使用OpenAI API进行请求
            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": summary_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens
            )
            # 将API返回的内容添加到输出列表中
            summary_output = response["choices"][0]["message"]["content"]

            for_output = for_output + str(ideas) + "\n\n"
            ideas.remove_entity(first_entity)
            ideas.remove_entity(second_entity)
            if first_entity in avescore and second_entity in avescore:
                del avescore[first_entity]
                del avescore[second_entity]
            
            ideas, new_entity_name = extract_entitiy_and_hints(ideas, summary_output)

            #print(str(ideas))

            score_prompt = construct_score_prompt(args, sample, ideas, new_entity_name)

            # 使用OpenAI API进行请求
            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": score_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens
            )
            # 将API返回的内容添加到输出列表中
            score_outputs = response["choices"][0]["message"]["content"]

            new_idea_scores_pair = extract_entity_scores(score_outputs, new_entity_name)
            new_idea_averscore = average_entity_scores(new_idea_scores_pair)        
            avescore[new_entity_name] = new_idea_averscore
            #print(avescore)

            if len(ideas.entities) == 1:
                final_hint.append(ideas.entities[new_entity_name])
                with open(output_for_file, 'w') as f:
                    f.write(for_output)
                break
    return final_hint


def last_hints_to_txt(samples, final_hint, out_file):
    last_dir = out_file + "/last"
    for sample, hint in zip(samples, final_hint):
        file_idx = sample['idx']
        last_dir = out_file +'/last'
        os.makedirs(last_dir, exist_ok=True)  
        output_last_hint_file = last_dir + f'/{file_idx}.txt'

        with open(output_last_hint_file, 'w') as f:
            f.write(hint.only_hint())


def read_last_entities(args, first, last, out_file, samples):
    hint_dir = out_file + '/last'
    final_dir = out_file + '/final_com'
    entity_dir = out_file + '/score'
    prompts = []
    entity_lists = []
    hint_lists = []
   
    for sample in tqdm(samples, total=len(samples), desc="Read and regen"):
        file_idx = int(sample['idx'])
        if file_idx >= first and file_idx <= last:
            if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub":
                code_dir = out_file + '/code'
                os.makedirs(code_dir, exist_ok=True)
            os.makedirs(hint_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            hint_file = hint_dir + f'/{file_idx}.txt'
            entity_file = entity_dir + f'/{file_idx}.txt'
            hints = read_and_print_file(hint_file)
            entities = read_and_print_file(entity_file)
            entity_list = extract_str_entity(entities)
            hint_list = extract_str_hints(hints)
            if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub":
                #prompt = construct_finalcode_prompt_byread(args, sample, entity_list, hint_list)
                prompt = construct_final_prompt_byread(args, sample, entity_list, hint_list)
            else:
                prompt = construct_final_prompt_byread(args, sample, entity_list, hint_list)
            prompts.append(prompt)
            entity_lists.append(entity_list)
            hint_lists.append(hint_list)

    return prompts, entity_lists, hint_lists



def only_read_entities(args, first, last, out_file, samples):
    entity_dir = out_file + '/score'
    prompts = []
    entity_lists = []
   
    for sample in tqdm(samples, total=len(samples), desc="Read and regen"):
        file_idx = int(sample['idx'])
        if file_idx >= first and file_idx <= last:
            if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub":
                code_dir = out_file + '/code'
                os.makedirs(code_dir, exist_ok=True)

            entity_file = entity_dir + f'/{file_idx}.txt'
            entities = read_and_print_file(entity_file)
            entity_list = extract_str_entity(entities)
            if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub":
                #prompt = construct_finalcode_prompt_byread(args, sample, entity_list, hint_list)
                prompt = construct_edd_prompt_byread(args, sample, entity_list)
            else:
                prompt = construct_edd_prompt_byread(args, sample, entity_list)
            prompts.append(prompt)
            entity_lists.append(entity_list)

    return prompts, entity_lists



def gpt_gen_math_final_byread(args, stop_tokens, out_file, samples, executor, first, last):
    final_outputs = []
    MAX_CODE_FIX_RETRIES = args.max_code_fix_retries
    num_samples = args.scnum
    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call
    prompts, entity_lists, hint_lists = read_last_entities(args, first, last, out_file, samples)

    for sample, prompt, entity_list, hint_list in tqdm(zip(samples, prompts, entity_lists, hint_lists), total=len(samples), desc="Final POT"):
        file_idx = sample['idx']
        final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final_code'
        code_output_dir = out_file + '/code' + f'/{file_idx}'
        os.makedirs(final_output_dir, exist_ok=True)
        os.makedirs(code_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += ', '.join(entity_list)
        final_answer_txt = '\n\n###Event Hints:' + '\n'.join(hint_list)
        final_answer_output += final_answer_txt

        self_consistency_answer = [] 
        self_consistency_thought = [] 

        for sc_epoch in range(num_samples):
            code_output_file = code_output_dir + f'/{sc_epoch}.txt'
            exec_result = ""
            print("=" * 50, "self-consistency epoch 1-5")
            for epoch in range(max_func_call):
                pass_key = False
                print("=" * 50, "Epoch", epoch)

    
                response = completion_with_backoff(
                    model=args.model_name_or_path,
                    messages=[{"role": "system", "content": initial_system_prompt},
                            {"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=args.temperature,
                    top_p=args.top_p,   
                    stop=stop_tokens
                )
                final_output = response["choices"][0]["message"]["content"]
                if "**Python Code**" in final_output:
                    code_output = extract_program(final_output)
                
                self_consistency_thought.append(final_output)
                code_result = executor.batch_apply(code_output)
                pred, report = code_result[0]
                pred, report = str(pred).strip(), str(report).strip()
                exec_result = pred if pred else report

                if exec_result == "":
                    exec_result += "<warning>\nDid you forget to use print()?\n</warning>\n"
                elif "Error" in exec_result:
                    # Split the query string
                    split_query = prompt.split("Tried Times: 0")

                    # Check if the split result has at least one element and if the last element is not empty
                    if split_query and split_query[-1]:
                        # Count the occurrences of the warning message in the last part of the split query
                        tried_times = split_query[-1].count("<warning>\nThe previous code block") + 1
                    else:
                        # If the split result is empty or the last element is empty, set tried_times to 0
                        tried_times = 0 
                    # Convert the integer tried_times to a string and append the warning message to exec_result
                    if tried_times <= (MAX_CODE_FIX_RETRIES - 1):
                        if args.verbose: print("Errors haven been occured.\n<extracted program>\n", code_output, "\n</extracted program>\n")
                        exec_result += "<warning>\nThe previous code block is not executable, will be removed from the code execution context. Please rewrite and fix this code block. (Tried Times: " + str(tried_times) + ")\n</warning>\n"
                        if args.code_concat and tried_times >= 1:
                            exec_result += "<current_full_code_context>\n" + code_output + "\n</current_full_code_context>\n"
                    else:
                        exec_result += "<warning>\nYou have tried to execute the code block " + str(tried_times) + " times, but it is still not executable. Please stop writing code for this question and try to solve this question manually.\n</warning>\nLet's think step by step, without using code. "
                else:
                    pass_key = True


                if pass_key == True:
                    with open(code_output_file, 'w') as f:
                        f.write(code_output)
                    break
                else:
                    if epoch == max_func_call - 2:
                        prompt += "\n<system>\nReach the max reponse limit, you must finish your reasoning and give your final solution in next reponse without resorting python code.\n</system>\n"
                    elif epoch == max_func_call - 1:
                        prompt += "\nReach max function call limit."
                    else:
                        prompt += exec_result
            
            match = re.search(r'\d+\.?\d*', exec_result)
            first_number = '0'
            if match:
                first_number = match.group()
                print("The first number found is:", first_number)
            else:
                print("No number found.")
            exec_result = float(first_number)
            self_consistency_answer.append(exec_result)
            #print(exec_result)
            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch] + "\n\nanswer:" + str(exec_result)




        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)

        # 将API返回的内容添加到输出列表中
        final_outputs.append(common_answer)
    return final_outputs



def gpt_gen_ehdd(args, stop_tokens, out_file, samples, remain_entities):
    final_outputs = []
    num_samples = args.scnum

    for sample, entity in tqdm(zip(samples, remain_entities), total=len(samples), desc="EHDD"):
        file_idx = sample['idx']
        if args.data_name == "AQUA" or args.data_name == "CSQA":
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nOptions:{sample['options']}\n\n\\\Ground:{sample['gt']}\n\nEntities: "
        else:
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final_com'
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += entity.output_entities()
        final_answer_txt = '\n\n###Event Hints:' + str(entity)
        final_answer_output += final_answer_txt

        final_prompt = construct_ehdd_prompt(args, sample, entity)
        self_consistency_answer = [] 
        self_consistency_thought = [] 

        for sc_epoch in range(num_samples):
            print("=" * 50, "self-consistency epoch 1-5")

            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": final_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,   
                stop=stop_tokens
            )
            final_output = response["choices"][0]["message"]["content"]
            
            self_consistency_thought.append(final_output)
            
            if args.data_name == "StrategyQA":
                match = re.search(r'\*\*Answer\(Yes or No\)\*\*\s*(Yes|No)', final_output)
                answer = "None"
            elif args.data_name == "AQUA" or args.data_name == "CSQA":
                match = re.search(r'\*\*Answer\(option choice\)\*\*\s*([A-E])', final_output)
                answer = "None"
            else:
                match = re.search(r'\*\*Answer\(arabic numerals\)\*\*\s*([\d\.]+)', final_output)
                answer = "None"
            if match:
                answer = match.group(1)  # 提取"Yes"或"No"
                print("Extracted answer:", answer)
            else:
                print("No answer found")
            self_consistency_answer.append(answer)
            #print(exec_result)

            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch]


        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)
        
        final_outputs.append(common_answer)
    return final_outputs






def gpt_gen_final(args, stop_tokens, out_file, samples, remain_entities, final_hint):
    final_outputs = []
    num_samples = args.scnum

    for sample, entity, hint in tqdm(zip(samples, remain_entities, final_hint), total=len(samples), desc="Final POT"):
        file_idx = sample['idx']
        if args.data_name == "AQUA" or args.data_name == "CSQA":
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nOptions:{sample['options']}\n\n\\\Ground:{sample['gt']}\n\nEntities: "
        else:
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final'
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += ', '.join(entity)
        final_answer_txt = '\n\n###Event Hints:' + hint.only_hint() 
        final_answer_output += final_answer_txt

        final_prompt = construct_final_prompt(args, sample, entity, hint)
        self_consistency_answer = [] 
        self_consistency_thought = [] 

        for sc_epoch in range(num_samples):
            print("=" * 50, "self-consistency epoch 1-5")

            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": final_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,   
                stop=stop_tokens
            )
            final_output = response["choices"][0]["message"]["content"]
            
            self_consistency_thought.append(final_output)
            
            if args.data_name == "StrategyQA":
                match = re.search(r'\*\*Answer\(Yes or No\)\*\*\s*(Yes|No)', final_output)
                answer = "None"
            elif args.data_name == "AQUA" or args.data_name == "CSQA":
                match = re.search(r'\*\*Answer\(option choice\)\*\*\s*([A-E])', final_output)
                answer = "None"
            else:
                match = re.search(r'\*\*Answer\(arabic numerals\)\*\*\s*([\d\.]+)', final_output)
                answer = "None"
            if match:
                answer = match.group(1)  # 提取"Yes"或"No"
                print("Extracted answer:", answer)
            else:
                print("No answer found")
            self_consistency_answer.append(answer)
            #print(exec_result)

            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch]


        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)
        
        final_outputs.append(common_answer)
    return final_outputs



def gpt_gen_final_byread(args, stop_tokens, out_file, samples, first, last):
    final_outputs = []
    MAX_CODE_FIX_RETRIES = args.max_code_fix_retries
    num_samples = args.scnum
    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call
    prompts, entity_lists, hint_lists = read_last_entities(args, first, last, out_file, samples)


    for sample, prompt, entity_list, hint_list in tqdm(zip(samples, prompts, entity_lists, hint_lists), total=len(samples), desc="Final"):
        file_idx = sample['idx']
        final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final_com'
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += ', '.join(entity_list)
        final_answer_txt = '\n\n###Event Hints:' + '\n'.join(hint_list)
        final_answer_output += final_answer_txt

        self_consistency_answer = [] 
        self_consistency_thought = [] 

        for sc_epoch in range(num_samples):
            print("=" * 50, "self-consistency epoch 1-5")

            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,   
                stop=stop_tokens
            )
            final_output = response["choices"][0]["message"]["content"]
            
            self_consistency_thought.append(final_output)
            
            
            if args.data_name == "StrategyQA":
                match = re.search(r'\*\*Answer\(Yes or No\)\*\*\s*(Yes|No)', final_output)
                answer = "None"
            elif args.data_name == "AQUA" or args.data_name == "CSQA":
                match = re.search(r'\*\*Answer\(option choice\)\*\*\s*([A-E])', final_output)
                answer = "None"
            else:
                match = re.search(r'\*\*Answer\(arabic numerals\)\*\*\s*([\d\.]+)', final_output)
                answer = "None"

            if match:
                answer = match.group(1)  # 提取"Yes"或"No"
                print("Extracted answer:", answer)
            else:
                print("No answer found")
            self_consistency_answer.append(answer)
            #print(exec_result)

            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch]
        
        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)
        
        final_outputs.append(common_answer)
    return final_outputs        







def gpt_gen_math_final(args, stop_tokens, out_file, samples, remain_entities, final_hint, executor):
    final_outputs = []
    MAX_CODE_FIX_RETRIES = args.max_code_fix_retries
    num_samples = args.scnum
    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call


    for sample, entity, hint in tqdm(zip(samples, remain_entities, final_hint), total=len(samples), desc="Final POT"):
        file_idx = sample['idx']
        final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final'
        code_output_dir = out_file + '/code' + f'/{file_idx}'
        os.makedirs(final_output_dir, exist_ok=True)
        os.makedirs(code_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += ', '.join(entity)
        final_answer_txt = '\n\n###Event Hints:' + hint.only_hint() 
        final_answer_output += final_answer_txt


        final_prompt = construct_finalcode_prompt(args, sample, entity, hint)
        self_consistency_answer = [] 
        self_consistency_thought = [] 

        for sc_epoch in range(num_samples):
            code_output_file = code_output_dir + f'/{sc_epoch}.txt'
            exec_result = ""
            print("=" * 50, "self-consistency epoch 1-5")
            for epoch in range(max_func_call):
                pass_key = False
                print("=" * 50, "Epoch", epoch)

    
                response = completion_with_backoff(
                    model=args.model_name_or_path,
                    messages=[{"role": "system", "content": initial_system_prompt},
                            {"role": "user", "content": final_prompt}],
                    max_tokens=2048,
                    temperature=args.temperature,
                    top_p=args.top_p,   
                    stop=stop_tokens
                )
                final_output = response["choices"][0]["message"]["content"]
                if "**Python Code**" in final_output:
                    code_output = extract_program(final_output)
                
                self_consistency_thought.append(final_output)
                code_result = executor.batch_apply(code_output)
                pred, report = code_result[0]
                pred, report = str(pred).strip(), str(report).strip()
                exec_result = pred if pred else report

                if exec_result == "":
                    exec_result += "<warning>\nDid you forget to use print()?\n</warning>\n"
                elif "Error" in exec_result:
                    # Split the query string
                    split_query = final_prompt.split("Tried Times: 0")

                    # Check if the split result has at least one element and if the last element is not empty
                    if split_query and split_query[-1]:
                        # Count the occurrences of the warning message in the last part of the split query
                        tried_times = split_query[-1].count("<warning>\nThe previous code block") + 1
                    else:
                        # If the split result is empty or the last element is empty, set tried_times to 0
                        tried_times = 0 
                    # Convert the integer tried_times to a string and append the warning message to exec_result
                    if tried_times <= (MAX_CODE_FIX_RETRIES - 1):
                        if args.verbose: print("Errors haven been occured.\n<extracted program>\n", code_output, "\n</extracted program>\n")
                        exec_result += "<warning>\nThe previous code block is not executable, will be removed from the code execution context. Please rewrite and fix this code block. (Tried Times: " + str(tried_times) + ")\n</warning>\n"
                        if args.code_concat and tried_times >= 1:
                            exec_result += "<current_full_code_context>\n" + code_output + "\n</current_full_code_context>\n"
                    else:
                        exec_result += "<warning>\nYou have tried to execute the code block " + str(tried_times) + " times, but it is still not executable. Please stop writing code for this question and try to solve this question manually.\n</warning>\nLet's think step by step, without using code. "
                else:
                    pass_key = True


                if pass_key == True:
                    with open(code_output_file, 'w') as f:
                        f.write(code_output)
                    break
                else:
                    if epoch == max_func_call - 2:
                        final_prompt += "\n<system>\nReach the max reponse limit, you must finish your reasoning and give your final solution in next reponse without resorting python code.\n</system>\n"
                    elif epoch == max_func_call - 1:
                        final_prompt += "\nReach max function call limit."
                    else:
                        final_prompt += exec_result
            
            match = re.search(r'\d+\.?\d*', exec_result)
            first_number = 0
            if match:
                first_number = int(match.group())
                print("The first number found is:", first_number)
            else:
                print("No number found.")
            exec_result = float(first_number)
            self_consistency_answer.append(exec_result)
            #print(exec_result)
            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch] + "\n\nanswer:" + str(exec_result)




        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)

        # 将API返回的内容添加到输出列表中
        final_outputs.append(common_answer)
    return final_outputs


def only_cot(args,stop_tokens, out_file, samples):
    final_outputs = []
    num_samples = args.scnum
    for sample in tqdm(samples, desc="Only COT"):  
        file_idx = sample['idx']
        if args.data_name == "AQUA" or args.data_name == "CSQA":
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nOptions:{sample['options']}\n\n\\\Ground:{sample['gt']}\n\n"
        else:
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\n"
        final_output_dir = out_file + '/final'
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'


        final_prompt = construct_prompt(args, sample)
        self_consistency_answer = [] 
        self_consistency_thought = []  

        for sc_epoch in range(num_samples):
            print("=" * 50, "self-consistency epoch 1-5")

            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": final_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,   
                stop=stop_tokens
            )
            final_output = response["choices"][0]["message"]["content"]
            
            self_consistency_thought.append(final_output)
            
            if args.data_name == "StrategyQA":
                match = re.search(r'\*\*Answer\(Yes or No\)\*\*\s*(Yes|No)', final_output)
                answer = "None"
            elif args.data_name == "AQUA" or args.data_name == "CSQA":
                match = re.search(r'\*\*Answer\(option choice\)\*\*\s*([A-E])', final_output)
                answer = "None"
            else:
                match = re.search(r'\*\*Answer\(arabic numerals\)\*\*\s*([-\d\.]+)', final_output)
                answer = "None"
            if match:
                answer = match.group(1)  # 提取"Yes"或"No"
                print("Extracted answer:", answer)
            else:
                print("No answer found")
            self_consistency_answer.append(answer)
            #print(exec_result)

            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch]


        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)
        
        final_outputs.append(common_answer)
    return final_outputs


#这个是只有实体的部分的消融实验
def gpt_edd(args,stop_tokens, out_file, samples, entities_outputs):
    final_outputs = []
    num_samples = args.scnum

    for sample, entity, final_prompt in tqdm(zip(samples, entities_outputs), total=len(samples), desc="Final EDD"):
        file_idx = sample['idx']
        if args.data_name == "AQUA" or args.data_name == "CSQA":
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nOptions:{sample['options']}\n\n\\\Ground:{sample['gt']}\n\nEntities: "
        else:
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final'
        os.makedirs(final_output_dir, exist_ok=True)
        str_entities = entity.output_entities()
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += str_entities
        final_prompt = construct_edd_prompt(args, sample, str_entities)

        self_consistency_answer = [] 
        self_consistency_thought = []  


        for sc_epoch in range(num_samples):
            print("=" * 50, "self-consistency epoch 1-5")

            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": final_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,   
                stop=stop_tokens
            )
            final_output = response["choices"][0]["message"]["content"]
            
            self_consistency_thought.append(final_output)
            
            if args.data_name == "StrategyQA":
                match = re.search(r'\*\*Answer\(Yes or No\)\*\*\s*(Yes|No)', final_output)
                answer = "None"
            elif args.data_name == "AQUA" or args.data_name == "CSQA":
                match = re.search(r'\*\*Answer\(option choice\)\*\*\s*([A-E])', final_output)
                answer = "None"
            else:
                match = re.search(r'\*\*Answer\(arabic numerals\)\*\*\s*([\d\.]+)', final_output)
                answer = "None"
            if match:
                answer = match.group(1)  # 提取"Yes"或"No"
                print("Extracted answer:", answer)
            else:
                print("No answer found")
            self_consistency_answer.append(answer)
            #print(exec_result)

            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch]


        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)
        
        final_outputs.append(common_answer)
    return final_outputs


def gpt_edd_read(args,stop_tokens, out_file, samples, first, last):
    final_outputs = []
    num_samples = args.scnum
    prompts, remain_entities = only_read_entities(args, first, last, out_file, samples)

    for sample, entity, final_prompt in tqdm(zip(samples, remain_entities, prompts), total=len(samples), desc="Final EDD"):
        file_idx = sample['idx']
        if args.data_name == "AQUA" or args.data_name == "CSQA":
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nOptions:{sample['options']}\n\n\\\Ground:{sample['gt']}\n\nEntities: "
        else:
            final_answer_output = f"idx:{file_idx}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "
        final_output_dir = out_file + '/final_com'
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_file = final_output_dir + f'/{file_idx}.txt'
        final_answer_output += ', '.join(entity)

        self_consistency_answer = [] 
        self_consistency_thought = []  


        for sc_epoch in range(num_samples):
            print("=" * 50, "self-consistency epoch 1-5")

            response = completion_with_backoff(
                model=args.model_name_or_path,
                messages=[{"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": final_prompt}],
                max_tokens=2048,
                temperature=args.temperature,
                top_p=args.top_p,   
                stop=stop_tokens
            )
            final_output = response["choices"][0]["message"]["content"]
            
            self_consistency_thought.append(final_output)
            
            if args.data_name == "StrategyQA":
                match = re.search(r'\*\*Answer\(Yes or No\)\*\*\s*(Yes|No)', final_output)
                answer = "None"
            elif args.data_name == "AQUA" or args.data_name == "CSQA":
                match = re.search(r'\*\*Answer\(option choice\)\*\*\s*([A-E])', final_output)
                answer = "None"
            else:
                match = re.search(r'\*\*Answer\(arabic numerals\)\*\*\s*([\d\.]+)', final_output)
                answer = "None"
            if match:
                answer = match.group(1)  # 提取"Yes"或"No"
                print("Extracted answer:", answer)
            else:
                print("No answer found")
            self_consistency_answer.append(answer)
            #print(exec_result)

            final_answer_output = final_answer_output + "\n\nself-consistency epoch:" + str(sc_epoch) + self_consistency_thought[sc_epoch]


        common_answer = aggregate_final_answer(self_consistency_answer)
        final_answer_output = final_answer_output + "\n\nfinal common answer:" + str(common_answer)
        with open(final_output_file, 'w') as f:
            f.write(final_answer_output)
        
        final_outputs.append(common_answer)
    return final_outputs


def is_string_not_convertible_to_float(var):
    # 检查变量是否是字符串类型
    if isinstance(var, str):
        try:
            # 尝试将字符串转换为浮点数
            float(var)
            # 如果转换成功，则返回False
            return False
        except ValueError:
            # 如果转换失败（抛出ValueError），则返回True
            return True
    else:
        # 如果变量不是字符串，直接返回False
        return False

def extract_letter(text):
    # 使用正则表达式查找第一个英文字母 A-Z 或 a-z
    match = re.search(r'[A-Za-z]', text)
    if match:
        return match.group(0).upper()  # 返回大写形式
    return None  # 没有找到字母则返回 None

def cal_acc(args, final_outputs, samples, out_file):
    correct = 0
    final_output_dir = out_file + '/final'
    final_true_output_dir = final_output_dir + '/true'
    final_false_output_dir = final_output_dir + '/false'
    os.makedirs(final_true_output_dir, exist_ok=True)
    os.makedirs(final_false_output_dir, exist_ok=True)

    for sample, answer in zip(samples, final_outputs):
        if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub" or args.data_name == "math":
            number_string = sample['gt'].replace(',', '')  # 删除逗号
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(number_string)
            print(answer_match)
            if answer == None:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            if args.data_name == "AddSub":
                integer_part = answer.split('.')[0]
                gt_part = number_string.split('.')[0]
                flag = is_string_not_convertible_to_float(answer)
                if flag == True:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
                elif float(integer_part) == float(gt_part):
                    correct += 1
                    destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
                else:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            else:
                integer_part = answer.split('.')[0]
                flag = is_string_not_convertible_to_float(answer)
                if flag == True:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
                elif float(integer_part) == float(number_string):
                    correct += 1
                    destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
                else:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
        
        elif args.data_name == "StrategyQA" or args.data_name == "AQUA" or args.data_name == "CSQA" or args.data_name == "bbh":
            if args.data_name == "StrategyQA" or args.data_name == "AQUA" or args.data_name == "CSQA":
                ground_answer = str(sample['gt'])
                source_path = final_output_dir + f"/{sample['idx']}.txt"
                answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(ground_answer)
                print(answer_match)
                if answer == None:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
                elif answer.lower() == ground_answer.lower():
                    correct += 1
                    destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
                else:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            else:
                ground_answer = str(sample['gt'])
                source_path = final_output_dir + f"/{sample['idx']}.txt"
                answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(ground_answer)
                print(answer_match)
                if answer == None:
                    destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
                else:
                    pred_letter = extract_letter(answer)
                    gt_letter = extract_letter(ground_answer)
                    if pred_letter.lower() == gt_letter.lower():
                        correct += 1
                        destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
                    else:
                        destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
                
        shutil.copy(source_path, destination_path)
        print(f"File copy from {source_path} to {destination_path}")



    acc = correct/len(samples)
    str_acc = "accuracy:" + str(acc)
    print(str_acc)

def validate_sub_questions(args, stop_tokens, original_question, sub_description, sub_answer):
    """
    验证单个子问题的合理性与答案一致性。

    Args:
        args: 参数对象
        stop_tokens: 生成用的 stop token 列表
        original_question: 原始问题（用于构造 prompt）
        sub_description: 子问题描述（字符串）
        sub_answer: 子问题答案（字符串）

    Returns:
        dict: 包含验证结果字段（统一格式，便于批量处理）
            - sub_problem_label: [原始描述]
            - sub_answer_label: [原始答案]
            - valid_problem_label: [重构反证问题]
            - valid_answer_label: [正向重构答案]
            - acc_label: ["1" 或 "0"]
            - analyze_solve_dict: {description: solve_analysis}，失败时记录分析
            - solve: 最后一次生成的分析文本（可选）
            - judge: 0 表示通过，1 表示失败
    """
    import re

    sub_problem_label = []
    sub_answer_label = []
    valid_problem_label = []
    valid_answer_label = []
    analyze_solve_dict = {}
    valid_answer = ''
    new_answer = ''
    solve = None

    print('本轮验证子问题:', sub_description)

    # === 1. 补全子问题 ===
    try:
        samples_complete = construct_complete_prompt(args, original_question, sub_description)
        completed_sub = gpt_gen_lot(args, stop_tokens, samples_complete)
        print('completed_sub',completed_sub)
        sub_text = re.search(r'<problem>(.*?)</problem>', completed_sub, re.DOTALL).group(1).strip()
    except (AttributeError, KeyError) as e:
        print(f"补全失败: {e}")



    # === 2. 重构子问题（正向推理）===
    try:
        samples_decompose = construct_backward_prompt(args, sub_text)
        without_delete_valid_sub = gpt_gen_lot(args, stop_tokens, samples_decompose)
        new_problem = re.search(r'<question>(.*?)</question>', without_delete_valid_sub, re.DOTALL).group(1).strip()
        new_answer = re.search(r'<answer>(.*?)</answer>', without_delete_valid_sub, re.DOTALL).group(1).strip()
    except (AttributeError, KeyError) as e:
        print(f"重构失败: {e}")
    print('complete:', without_delete_valid_sub)
    # === 3. 替换子问题为答案 ===
    try:
        samples_delete = construct_replace_prompt(args, new_problem, sub_answer)
        x_valid_sub = gpt_gen_lot(args, stop_tokens, samples_delete)
        x_sub_valid_question = re.search(r'<question>(.*?)</question>', x_valid_sub, re.DOTALL).group(1).strip()
    except (AttributeError, KeyError) as e:
        print(f"替换失败: {e}")
    # print('replace:' ,x_valid_sub)

    # === 4. 反向推理验证 ===
    if not x_sub_valid_question:
        print("反证问题为空，跳过验证")
        # 尝试生成分析结论
        for attempt in range(3):
            try:
                conclusion_prompt = construct_conclusion_prompt(args, original_question, sub_description)
                conclusion = gpt_gen_lot(args, stop_tokens, conclusion_prompt)
                solve = re.search(r'<solve>(.*?)</solve>', conclusion, re.DOTALL).group(1).strip()
                analyze_solve_dict[sub_description] = solve
                break
            except AttributeError:
                print(f"反证问题验证失败: {e}")
                continue

    try:
        first_forward_prompt = construct_valid_prompt(args, x_sub_valid_question)
        content_first = gpt_gen_lot(args, stop_tokens, first_forward_prompt)
        valid_answer = re.search(r'<answer>(.*?)</answer>', content_first, re.DOTALL).group(1).strip()
    except (AttributeError, KeyError):
        print("反向推理答案提取失败")
        # 尝试生成分析结论
        for attempt in range(3):
            try:
                conclusion_prompt = construct_conclusion_prompt(args, original_question, sub_description)
                conclusion = gpt_gen_lot(args, stop_tokens, conclusion_prompt)
                solve = re.search(r'<solve>(.*?)</solve>', conclusion, re.DOTALL).group(1).strip()
                analyze_solve_dict[sub_description] = solve
                break
            except AttributeError:
                print(f"反证问题总结失败")
                continue

    # === 6. 比较答案 ===
    valid_clean = valid_answer.split('.')[0].strip()
    new_clean = new_answer.split('.')[0].strip()

    if valid_clean == new_clean:
        print('valid_clean',valid_clean)
        print('new_clean',new_clean)
        label = 1
    else:
        print('valid_clean',valid_clean)
        print('new_clean',new_clean)
        label = 0
        # 生成分析结论
        for attempt in range(3):
            try:
                conclusion_prompt = construct_conclusion_prompt(args, original_question, sub_description)
                conclusion = gpt_gen_lot(args, stop_tokens, conclusion_prompt)
                solve = re.search(r'<solve>(.*?)</solve>', conclusion, re.DOTALL).group(1).strip()
                analyze_solve_dict[sub_description] = solve
                break
            except AttributeError:
                continue

    sub_problem_label.append(sub_description)
    sub_answer_label.append(sub_answer)
    valid_problem_label.append(x_sub_valid_question)
    valid_answer_label.append(new_answer)

    return {
        'sub_problem_label': sub_problem_label,
        'sub_answer_label': sub_answer_label,
        'valid_problem_label': valid_problem_label,
        'valid_answer_label': valid_answer_label,
        'acc_label': label,
        'analyze_solve_dict': analyze_solve_dict,
        'solve': solve,
    }


def main(args):
    # 加载数据集
    examples = load_data(args.data_name, args.split)  # 根据数据名称和数据分割类型加载数据

    # 如果指定了测试样本数量，从加载的样本中随机选择指定数量的样本，我们这里默认是全部测试，也可以部分测试
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)

    # 如果设置了随机打乱数据的选项,这里默认不打乱，所以这里一般跳过
    if args.shuffle:
        random.seed(datetime.now().timestamp())  # 设置随机种子为当前时间戳
        random.shuffle(examples)  # 打乱样本顺序

    # 如果结束点设置为-1，则使用样本总数作为结束点
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start:args.end]  # 根据起始和结束点切分样本集
 


    model_name = "/".join(args.model_name_or_path.split("/")[-2:])  # 从模型路径中提取模型名称
    file_prompt_type = args.prompt_type.replace("program_only", "tora")  # 替换prompt类型名称中的某个部分
    out_file_prefix = f'{args.split}_{file_prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'  # 构建输出文件前缀
    #out_file = f'outputs/{model_name}/{args.data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'  # 完整的输出文件路径
    out_file = f'outputs/{model_name}/{args.prompt_type}/{args.data_name}'
    os.makedirs(f'outputs/{model_name}/{args.prompt_type}/{args.data_name}', exist_ok=True)  # 创建输出目录，如果已存在则忽略
    # out_file = f'outputs/{model_name}/{args.data_name}'
    # os.makedirs(f'outputs/{model_name}/{args.data_name}', exist_ok=True)  # 创建输出目录，如果已存在则忽略

    processed_files = [f for f in os.listdir(f"outputs/{model_name}/{args.prompt_type}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]#这里是建立一个list，用来先列举在相同大模型环境下在用相同策略的情况下使用的结果
    processed_samples = []
    for f in processed_files:
        if args.prompt_type not in ["dd"] and (args.model_name_or_path in ["gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview"] or "gpt-4" in args.model_name_or_path or "gpt-3.5-turbo" in args.model_name_or_path):
            processed_samples.extend(list(load_jsonl(f"outputs/{model_name}/{args.prompt_type}/{args.data_name}/{f}")))
            print("f:", f)  # 打印处理过的文件名
        else:
            continue

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)


    # 去重处理
    processed_samples = {sample['idx']: sample for sample in processed_samples}  # 将已处理的样本列表转换成字典，使用样本的 'idx' 作为键，样本本身作为值，自动去重
    processed_idxs = list(processed_samples.keys())  # 获取已处理样本的索引列表
    processed_samples = list(processed_samples.values())  # 从字典中提取样本数据列表
    total_examples = len(examples)  # 计算当前批次总样本数
    if args.data_name == "AddSub":
        examples = [example for example in examples if example['qid'] not in processed_idxs]  # 筛选出未处理的样本
    elif args.data_name == "aime":
        examples = [example for example in examples if example['index'] not in processed_idxs]  # 筛选出未处理的样本
    elif args.data_name == "mmlu-pro":
        examples = [example for example in examples if example['question_id'] not in processed_idxs]  # 筛选出未处理的样本
    else:
        examples = [example for example in examples if example['idx'] not in processed_idxs]  # 筛选出未处理的样本
    print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")  # 打印当前处理的样本范围及剩余未处理的样本数量

    # 判断是否还有未处理的样本
    if len(examples) == 0:
        print("No examples to process.")  # 如果没有未处理的样本，打印消息
        pass  # 这里的 pass 是冗余的，可以省略
        # return  # 如果需要终止函数执行，可以取消注释这行
    else:
        print(examples[0])  # 如果有未处理的样本，打印第一个样本的信息

    samples = []

    for example in tqdm(examples, total=len(examples)):

        if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub" or args.data_name == "AQUA" or args.data_name == "bbh" or args.data_name == "math" or args.data_name == "lot" or args.data_name == "CommonsenseQA" or args.data_name == "MultiArith" or args.data_name == "aime" or args.data_name == "SVBench":
            if args.data_name == "AddSub":
                idx = example['qid']
            elif args.data_name == "aime":
                idx = example['index']
            else:
                idx = example['idx']


            # parse question and answer
            example['question'] = parse_question(example, args.data_name)
            gt_cot, gt_ans = parse_math_ground_truth(example, args.data_name)
            full_prompt = construct_first_prompt(args, example)#这里使用cr会使具有对应数据集的md文件添加进入prompt中
            if args.data_name == "MultiArith":
                sample = {'idx': idx, 'question': example['question'], 'gt': gt_ans}
            if args.data_name == "aime":
                sample = {'idx': idx, 'question': example['question'], 'gt': gt_ans}
            if args.data_name == "gsm8k" or args.data_name == "svamp" or args.data_name == "AddSub" or args.data_name == "bbh" or args.data_name == "math":
                sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}
            if args.data_name == "AQUA":
                sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'options': example['options'], 'gt': gt_ans, 'prompt': full_prompt}
            if args.data_name == "lot":
                if args.prompt_type == "cot":
                    sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'answer': example['answer']}
                else:
                    sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'answer': example['answer'], 'valid_question': example['valid_question'], 'valid_answer': example['valid_answer']}
            if args.data_name == "SVBench":
                sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'model_result': example["model_result"]}
            # add remain fields
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in example:
                    sample[key] = example[key]
            samples.append(sample)  
        elif args.data_name == "StrategyQA" or args.data_name == "CSQA":
            idx = example['idx']
            example['question'] = parse_question(example, args.data_name)
            gt_ans, gt_cot = parse_sqa_ground_truth(example)
            # full_prompt = construct_first_prompt(args, example['question'])
            if args.data_name == "StrategyQA":
                sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}
            else:
                sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'options': example['options'], 'gt': gt_ans, 'concept': example['concept']}
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in example:
                    sample[key] = example[key]
            samples.append(sample)  

        elif args.data_name == "mmlu-pro":
            idx = example['question_id']
            example['question'] = parse_question(example, args.data_name)
            gt_cot, gt_ans = parse_math_ground_truth(example, args.data_name)
            sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'options': example['options']}
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in example:
                    sample[key] = example[key]
            samples.append(sample)


    print("dataset:", args.data_name, "samples:", len(samples))

    if len(samples) > 0:
        print("-" * 50)
        print("sample:")
        print("-" * 50)

    # repeat n times

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call
    stop_tokens = ["</s>", "---", "```output"]
    stop_conclusion = ["</s>"]
               
    if args.prompt_type == "cot":
        final_outputs = []
        # print('循环对了,就是cot')
        for sample in tqdm(samples, desc="Get first result:"):
            # print('现在的题目是', sample['question'])
            cot_label = []
            final_answer = ""
            d_infer_answer = ""
            #直接推理
            first_forward_prompt = construct_cot_prompt(args, sample)
            print('first_forward_prompt', first_forward_prompt)
            content_first = gpt_gen_lot(args, stop_tokens, first_forward_prompt)
            print('content_first', content_first)
            for attempt in range(3):
                try:
                    d_infer_answer = re.search(r'<answer>(.*?)</answer>', content_first, re.DOTALL).group(1)
                except AttributeError:
                    break
            final_answer = d_infer_answer
            cot_label.append(content_first)
            final_outputs.append(final_answer)
            # print('现在开始保存')
            file_idx = sample['idx']
            name_dataset = args.name_of_dataset
            final_output_dir = out_file + '/final'
            final_output_file = final_output_dir + f'/{file_idx}.txt'
            # print('保存地址',final_output_file)
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            final_answer_output = (
                f"idx:{file_idx}\n\n"
                f"Question:{sample['question']}\n\n"
                f"Ground:{sample['gt']}\n\n"
            )
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(final_answer_output + '\n\n')
                f.write("COT:\n")
                f.write('\n'.join(cot_label) + '\n\n')
                f.write("final answer:" + final_answer + "\n")

    elif args.prompt_type == "judge":
        final_outputs = []
        for sample in tqdm(samples, desc="Processing samples:"):
            # --- 1. 简单 CoT 推理 (初始测试) ---
            # 重命名变量以避免与后续验证结果冲突
            initial_cot_content = ""
            initial_cot_answer = "Extract Failed"
            
            # 直接推理
            # print('args.name_of_dataset:', args.name_of_dataset)
            first_forward_prompt = construct_cot_prompt(args, sample)
            print('first_forward_prompt', first_forward_prompt)
            initial_cot_content = gpt_gen_lot(args, stop_tokens, first_forward_prompt)
            print('content_first', initial_cot_content)
            
            # 提取初始 CoT 答案
            for attempt in range(3):
                try:
                    match = re.search(r'<answer>(.*?)</answer>', initial_cot_content, re.DOTALL)
                    if match:
                        initial_cot_answer = match.group(1)
                        break
                except AttributeError:
                    pass
            
            # --- 2. 模型验证循环 ---
            models_info_list = [] 
            file_idx = sample['idx']
            
            # 【内层循环】遍历 3 个模型的结果，对每个调用 API 进行验证
            for model_idx, model_item in enumerate(sample['model_result']):
                judge_answer = ""  # 重命名变量，避免冲突
                
                # 从字典中获取模型的 cot 内容（用于验证）
                # model_cot = model_item.get('cot', '') 
                
                # 从数据集中获取该模型的验证正确性标签
                is_correct = model_item.get('is_correct', 'Unknown')
                model_name = model_item.get('model_name', 'Unknown')
                original_answer = model_item.get('final_answer', '')
                
                # 【构造验证 prompt】- 传入 sample 和模型的 cot 内容
                judge_prompt = construct_judge_prompt(args, model_item, sample['question'])
                print(f'Model {model_idx + 1} ({model_name}) judge_prompt:', judge_prompt)
                
                # 【调用 API 进行验证】
                content_judge = gpt_gen_lot(args, stop_tokens, judge_prompt)
                print(f'Model {model_idx + 1} ({model_name}) judge result:', content_judge)
                
                # 提取验证结果中的 answer
                for attempt in range(3):
                    try:
                        match = re.search(r'<judge>(.*?)</judge>', content_judge, re.DOTALL)
                        if match:
                            judge_answer = match.group(1)
                            break
                    except AttributeError:
                        judge_answer = "Extract Failed"
                        break
                
                models_info_list.append({
                    "model_id": model_idx + 1,
                    "model_name": model_name,
                    # "cot_content": model_cot,
                    "original_final_answer": original_answer,
                    "judge_final_answer": judge_answer,
                    "is_correct": is_correct,  # 来自数据集的验证标签
                    "judge_output": content_judge
                })

            # --- 3. 保存逻辑 ---
            # 三个模型结果保存在一个 txt 文件里
            final_output_dir = out_file + '/final'
            final_output_file = final_output_dir + f'/{file_idx}.txt'
            
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            
            # 构建文件内容
            file_content = (
                f"idx:{file_idx}\n\n"
                f"Question:{sample['question']}\n\n"
                f"Ground Truth:{sample['gt']}\n\n"
                f"{'='*50}\n"
                f"Part 1: Initial Simple CoT Test\n"
                f"{'='*50}\n\n"
                f"CoT Content:\n{initial_cot_content}\n\n"
                f"Extracted Answer: {initial_cot_answer}\n\n"
                f"{'='*50}\n"
                f"Part 2: Model Results Comparison (3 Models)\n"
                f"{'='*50}\n\n"
            )
            
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
                
                for info in models_info_list:
                    f.write(f"{'='*50}\n")
                    f.write(f"Model {info['model_id']}: {info['model_name']}\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Is Correct (Dataset Label): {info['is_correct']}\n\n")
                    # f.write("Model Output (COT):\n")
                    # f.write(info['cot_content'] + '\n\n')
                    f.write(f"Original Final Answer: {info['original_final_answer']}\n")
                    f.write(f"Judge Final Answer: {info['judge_final_answer']}\n")
                    f.write(f"Judge Full Output:\n{info['judge_output']}\n")
                    f.write("\n" + "-"*50 + "\n\n")
            
            # 记录结果到列表 (保持原有逻辑，仅记录验证结果)
            final_outputs.append([info['judge_final_answer'] for info in models_info_list])


    elif args.prompt_type == "judge-cot":
        final_outputs = []
        for sample in tqdm(samples, desc="Processing samples:"):
            
            # --- 2. 模型验证循环 ---
            models_info_list = [] 
            file_idx = sample['idx']
            
            # 【内层循环】遍历 3 个模型的结果，对每个调用 API 进行验证
            for model_idx, model_item in enumerate(sample['model_result']):
                judge_answer = ""  # 重命名变量，避免冲突
                
                # 从字典中获取模型的 cot 内容（用于验证）
                # model_cot = model_item.get('cot', '') 
                
                # 从数据集中获取该模型的验证正确性标签
                is_correct = model_item.get('is_correct', 'Unknown')
                model_name = model_item.get('model_name', 'Unknown')
                original_answer = model_item.get('final_answer', '')
                
                # 【构造验证 prompt】- 传入 sample 和模型的 cot 内容
                judge_prompt = construct_judge_prompt(args, model_item, sample['question'])
                print(f'Model {model_idx + 1} ({model_name}) judge_prompt:', judge_prompt)
                
                # 【调用 API 进行验证】
                content_judge = gpt_gen_lot(args, stop_tokens, judge_prompt)
                print(f'Model {model_idx + 1} ({model_name}) judge result:', content_judge)
                
                # 提取验证结果中的 answer
                for attempt in range(3):
                    try:
                        match = re.search(r'<judge>(.*?)</judge>', content_judge, re.DOTALL)
                        if match:
                            judge_answer = match.group(1)
                            break
                    except AttributeError:
                        judge_answer = "Extract Failed"
                        break
                
                models_info_list.append({
                    "model_id": model_idx + 1,
                    "model_name": model_name,
                    # "cot_content": model_cot,
                    "original_final_answer": original_answer,
                    "judge_final_answer": judge_answer,
                    "is_correct": is_correct,  # 来自数据集的验证标签
                    "judge_output": content_judge
                })

            # --- 3. 保存逻辑 ---
            # 三个模型结果保存在一个 txt 文件里
            final_output_dir = out_file + '/final'
            final_output_file = final_output_dir + f'/{file_idx}.txt'
            
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            
            # 构建文件内容
            file_content = (
                f"idx:{file_idx}\n\n"
                f"Question:{sample['question']}\n\n"
                f"Ground Truth:{sample['gt']}\n\n"
                f"{'='*50}\n"
                # f"Part 1: Initial Simple CoT Test\n"
                # f"{'='*50}\n\n"
                # f"CoT Content:\n{initial_cot_content}\n\n"
                # f"Extracted Answer: {initial_cot_answer}\n\n"
                f"{'='*50}\n"
                f"Part 2: Model Results Comparison (3 Models)\n"
                f"{'='*50}\n\n"
            )
            
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
                
                for info in models_info_list:
                    f.write(f"{'='*50}\n")
                    f.write(f"Model {info['model_id']}: {info['model_name']}\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Is Correct (Dataset Label): {info['is_correct']}\n\n")
                    # f.write("Model Output (COT):\n")
                    # f.write(info['cot_content'] + '\n\n')
                    f.write(f"Original Final Answer: {info['original_final_answer']}\n")
                    f.write(f"Judge Final Answer: {info['judge_final_answer']}\n")
                    f.write(f"Judge Full Output:\n{info['judge_output']}\n")
                    f.write("\n" + "-"*50 + "\n\n")
            
            # 记录结果到列表 (保持原有逻辑，仅记录验证结果)
            final_outputs.append([info['judge_final_answer'] for info in models_info_list])

    elif args.prompt_type == "judge-location":
        final_outputs = []
        # 确保输出目录存在
        os.makedirs(out_file, exist_ok=True)
        os.makedirs(os.path.join(out_file, 'final'), exist_ok=True)
        
        for sample in tqdm(samples, desc="Processing samples:"):
            models_info_list = []
            file_idx = sample['idx']

            # 遍历 3 个模型的结果
            for model_idx, model_item in enumerate(sample['model_result']):
                # 初始化每个模型的信息字典
                model_info = {
                    'model_name': model_item.get('model_name', 'Unknown'),
                    'is_correct': model_item.get('is_correct', 'Unknown'),
                    'original_final_answer': model_item.get('final_answer', ''),
                    'error_reason': "N/A",
                    'error_judge': "N/A",
                    'judge_output': "N/A"
                }

                # 获取模型结果信息
                is_correct = model_item.get('is_correct', 'Unknown')

                # 仅对判断结果为错误的情况执行 API 请求
                # 兼容布尔值 False 或字符串 'false'
                if is_correct is False or str(is_correct).lower() == 'false':
                    # 构造错误分析 prompt
                    judge_prompt = construct_reason_prompt(args, model_item, sample['question'])

                    # 调用 API 进行错误原因分析
                    content_judge = gpt_gen_lot(args, stop_tokens, judge_prompt)

                    # 提取分析结果
                    error_reason = "Extract Failed"
                    for attempt in range(3):
                        try:
                            match = re.search(r'<reason>(.*?)</reason>', content_judge, re.DOTALL)
                            if match:
                                error_reason = match.group(1).strip()
                                break
                        except AttributeError:
                            error_reason = "Extract Failed"
                            break

                    model_info['judge_output'] = content_judge
                    model_info['error_reason'] = error_reason

                    judge_prompt_2 = construct_reason_judge_prompt(args, model_item, sample['question'], error_reason)
                    content_judge_2 = gpt_gen_lot_2(args, stop_tokens, judge_prompt_2)

                    error_judge = "Extract Failed"
                    for attempt in range(3):
                        try:
                            match = re.search(r'<judge>(.*?)</judge>', content_judge_2, re.DOTALL)
                            if match:
                                error_judge = match.group(1).strip()
                                break
                        except AttributeError:
                            error_judge = "Extract Failed"
                            break

                    model_info['error_judge'] = error_judge

                # 将当前模型的信息添加到列表中
                models_info_list.append(model_info)
            final_output_dir = out_file + '/final'
            final_output_file = final_output_dir + f'/{file_idx}.txt'
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(f"idx:{file_idx}\n\n")
                f.write(f"Question:{sample['question']}\n\n")
                f.write(f"Ground Truth:{sample['gt']}\n\n")
                f.write(f"{'='*50}\n")
                f.write(f"Model Error Analysis Results\n")
                f.write(f"{'='*50}\n\n")
                
                for info in models_info_list:
                    f.write(f"Model: {info['model_name']}\n")
                    f.write(f"Is Correct: {info['is_correct']}\n")
                    f.write(f"Original Answer: {info['original_final_answer']}\n")
                    f.write(f"Error Reason:\n{info['error_reason']}\n")
                    f.write(f"Error Judge:\n{info['error_judge']}\n")
                    if info['judge_output'] != "N/A":
                        f.write(f"Full Analysis Output:\n{info['judge_output']}\n")
                    f.write("\n" + "-"*50 + "\n\n")
            
            # 记录结果到列表
            error_reasons = [info['error_reason'] for info in models_info_list if info['error_reason'] != "N/A" and info['error_reason'] != "N/A (Correct)"]
            final_outputs.append(error_reasons)

        
        print(f"Processing complete. TXT logs saved to {out_file}/final/")


    elif args.prompt_type == "reason":
        final_outputs = []
        # 确保输出目录存在
        os.makedirs(out_file, exist_ok=True)
        os.makedirs(os.path.join(out_file, 'final'), exist_ok=True)
        
        for sample in tqdm(samples, desc="Processing samples:"):
            models_info_list = [] 
            file_idx = sample['idx']
            
            # 遍历 3 个模型的结果
            for model_idx, model_item in enumerate(sample['model_result']):
                error_reason = "N/A"
                judge_output = "N/A"
                
                # 获取模型结果信息
                is_correct = model_item.get('is_correct', 'Unknown')
                model_name = model_item.get('model_name', 'Unknown')
                original_answer = model_item.get('final_answer', '')
                
                # 仅对判断结果为错误的情况执行 API 请求
                # 兼容布尔值 False 或字符串 'false'
                if is_correct is False or str(is_correct).lower() == 'false':
                    # 构造错误分析 prompt
                    judge_prompt = construct_reason_prompt(args, model_item, sample['question'])
                    
                    # 调用 API 进行错误原因分析
                    content_judge = gpt_gen_lot(args, stop_tokens, judge_prompt)
                    
                    # 提取分析结果
                    for attempt in range(3):
                        try:
                            match = re.search(r'<reason>(.*?)</reason>', content_judge, re.DOTALL)
                            if match:
                                error_reason = match.group(1)
                                break
                        except AttributeError:
                            error_reason = "Extract Failed"
                            break
                    
                    judge_output = content_judge
                    
                    # 【关键】将分析结果写入当前 sample 对象的内存中
                    sample['model_result'][model_idx]['error_reason'] = error_reason
                    sample['model_result'][model_idx]['judge_output'] = judge_output
                else:
                    # 如果正确，标记为无需分析
                    sample['model_result'][model_idx]['error_reason'] = "N/A (Correct)"

                models_info_list.append({
                    "model_id": model_idx + 1,
                    "model_name": model_name,
                    "original_final_answer": original_answer,
                    "is_correct": is_correct,
                    "error_reason": error_reason,
                    "judge_output": judge_output
                })

            # --- 保存 TXT 日志文件 (方便查看具体错误) ---
            final_output_file = os.path.join(out_file, 'final', f'{file_idx}.txt')
            
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(f"idx:{file_idx}\n\n")
                f.write(f"Question:{sample['question']}\n\n")
                f.write(f"Ground Truth:{sample['gt']}\n\n")
                f.write(f"{'='*50}\n")
                f.write(f"Model Error Analysis Results\n")
                f.write(f"{'='*50}\n\n")
                
                for info in models_info_list:
                    f.write(f"Model: {info['model_name']}\n")
                    f.write(f"Is Correct: {info['is_correct']}\n")
                    f.write(f"Original Answer: {info['original_final_answer']}\n")
                    f.write(f"Error Reason:\n{info['error_reason']}\n")
                    if info['judge_output'] != "N/A":
                        f.write(f"Full Analysis Output:\n{info['judge_output']}\n")
                    f.write("\n" + "-"*50 + "\n\n")
            
            # 记录结果到列表
            error_reasons = [info['error_reason'] for info in models_info_list if info['error_reason'] != "N/A" and info['error_reason'] != "N/A (Correct)"]
            final_outputs.append(error_reasons)

        # --- 【新增】将所有修改后的样本保存为新的 JSONL 文件 ---
        # 这样新的字段 'error_reason' 才会持久化保存
        output_jsonl_path = os.path.join(out_file, 'updated_samples.jsonl')
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Processing complete. TXT logs saved to {out_file}/final/")
        print(f"Updated dataset with error_reason saved to {output_jsonl_path}")


    elif args.prompt_type == "sccot":
        final_outputs = []
        self_consistency_answer = []
        print('循环对了,就是cot')
        for sample in tqdm(samples, desc="Get first result:"):
            print('现在的题目是', sample['question'])
            cot_label = []
            final_answer = ""
            for a in range(args.scnum):
            #直接推理
                for attempt in range(3):
                    try:
                        first_forward_prompt = construct_first_prompt(args, sample)
                        print('first_forward_prompt', first_forward_prompt)
                        content_first = gpt_gen_lot(args, stop_tokens, first_forward_prompt)
                        print('content_first', content_first)
                        d_infer_answer = re.search(r'<answer>(.*?)</answer>', content_first, re.DOTALL).group(1)
                        break
                    except AttributeError as e:
                        print(f"first...AttributeError:{e}")
                cot_label.append(content_first)
                self_consistency_answer.append(d_infer_answer)
            final_answer = aggregate_final_answer(self_consistency_answer)
            final_outputs.append(final_answer)
            # print('现在开始保存')
            file_idx = sample['idx']
            name_dataset = args.name_of_dataset
            final_output_dir = out_file + '/final'
            final_output_file = final_output_dir + f'/{file_idx}.txt'
            # print('保存地址',final_output_file)
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            final_answer_output = (
                f"idx:{file_idx}\n\n"
                f"Question:{sample['question']}\n\n"
                f"Ground:{sample['gt']}\n\n"
            )
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(final_answer_output + '\n\n')
                f.write("COT:\n")
                f.write('\n'.join(cot_label) + '\n\n')
                f.write("final answer:" + final_answer + "\n")

    elif args.prompt_type == "temp":
        final_outputs = only_cot(args, stop_tokens, out_file, samples)
    elif args.prompt_type == "comcot":
        final_outputs = only_cot(args, stop_tokens, out_file, samples)
    elif args.prompt_type == "sccomcot":
        final_outputs = only_cot(args, stop_tokens, out_file, samples)

    cal_acc(args, final_outputs, samples, out_file)






if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)