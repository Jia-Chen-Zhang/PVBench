import os
import re
import glob
import json

def calculate_accuracies(folder_path, dataset_path=None, output_jsonl_path=None):
    # 统计变量
    total_files = 0
    initial_correct_count = 0
    total_naive_verifications = 0
    correct_naive_verifications = 0
    correct_advanced_count = 0
    total_score = 0  # 新增：总分统计（对+1分，错-1分）
    scored_files = 0  # 新增：计入总分的文件数（所有文件）

    # 记录三类文件的 ID
    incomplete_files = []   # 验证捕获失败/不完整 (<3次验证)
    success_files = []      # 三次验证均正确
    failed_files = []       # 新增：有3次验证但至少一次错误

    # 存储不完整文件的完整数据（用于输出到jsonl）
    incomplete_data = []

    # 新增细粒度统计变量
    true_label_total = 0            # 标签为"True"的总次数
    false_label_total = 0           # 标签为"False"的总次数
    true_label_judged_false = 0    # 标签为"True"但被判断为"false"的次数
    false_label_judged_true = 0    # 标签为"False"但被判断为"true"的次数

    # 预编译正则
    gt_pattern = re.compile(r"Ground Truth:\s*(.*)", re.IGNORECASE)
    ext_ans_pattern = re.compile(r"Extracted Answer:\s*(.*)", re.IGNORECASE)
    judge_pattern = re.compile(
        r"Is Correct \(Dataset Label\):\s*(True|False).*?Judge Final Answer:\s*(true|false)",
        re.IGNORECASE | re.DOTALL
    )

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for filepath in txt_files:
        total_files += 1
        filename = os.path.basename(filepath)

        # 1. 提取文件 ID (如 "1206.txt" -> "1206")
        idx_match = re.search(r'(\d+)', filename)
        idx = idx_match.group(1) if idx_match else None

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. 初始准确率
        gt_match = gt_pattern.search(content)
        ext_ans_match = ext_ans_pattern.search(content)
        if gt_match and ext_ans_match:
            if gt_match.group(1).strip() == ext_ans_match.group(1).strip():
                initial_correct_count += 1

        # 3. 验证准确率
        judges = judge_pattern.findall(content)
        file_correct_count = 0

        for label, answer in judges:
            total_naive_verifications += 1

            # 统一处理大小写
            label_lower = label.strip().lower()
            answer_lower = answer.strip().lower()

            # 统计标签类型和细粒度错误
            if label_lower == "true":
                true_label_total += 1
                if answer_lower == "false":
                    true_label_judged_false += 1
            elif label_lower == "false":
                false_label_total += 1
                if answer_lower == "true":
                    false_label_judged_true += 1

            # 原有的正确性判断保持不变
            if label_lower == answer_lower:
                correct_naive_verifications += 1
                file_correct_count += 1

        # 4. 分类记录和分数统计
        if idx:
            if len(judges) != 3:
                # 捕获不完整
                incomplete_files.append(idx)

                # 读取并保存原始文件内容
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # 构建不完整文件的数据
                incomplete_data.append({
                    "id": idx,
                    "filename": filename,
                    "filepath": filepath,
                    "content": file_content,
                    "judges_count": len(judges),
                    "judges": judges
                })

                # 所有文件都计入评分分母，不完整验证文件得0分
                scored_files += 1
                total_score += 0  # 得0分

            elif len(judges) == 3:
                # 有完整3次验证，进行分数统计
                scored_files += 1
                if file_correct_count == 3:
                    # 三次验证均正确
                    success_files.append(idx)
                    correct_advanced_count += 1
                    total_score += 1  # 得+1分
                else:
                    # 至少一次错误
                    failed_files.append(idx)
                    total_score -= 1  # 得-1分

    # 输出统计
    expected_verifications = total_files * 3  # 每个文件应该有3次验证

    print("\n" + "="*50)
    print("📊 测评结果统计报告")
    print("="*50)
    print(f"总处理文件数：{total_files}")
    print(f"预期验证总次数：{expected_verifications}")
    print(f"实际匹配到的验证次数：{total_naive_verifications}\n")

    # 1. 开头结果准确率
    initial_acc = (initial_correct_count / total_files) * 100 if total_files > 0 else 0
    print(f"1️⃣ 开头结果准确率：{initial_correct_count}/{total_files} ({initial_acc:.2f}%)\n")

    # 2. 朴素验证准确率（分母为总文件数 * 3）
    naive_acc = (correct_naive_verifications / expected_verifications) * 100 if expected_verifications > 0 else 0
    print(f"2️⃣ 朴素验证准确率：{correct_naive_verifications}/{expected_verifications} ({naive_acc:.2f}%)")
    print(f"   匹配正确率：{correct_naive_verifications}/{total_naive_verifications} ({(correct_naive_verifications/total_naive_verifications*100):.2f}%)\n")

    # 3. 细粒度错误分析（新增）
    print(f"🔍 细粒度错误分析")
    print("-" * 30)

    # 计算原本True被判断错误率
    true_misjudge_rate = 0
    if true_label_total > 0:
        true_misjudge_rate = (true_label_judged_false / true_label_total) * 100

    # 计算原本False判断正确率
    false_correct_rate = 0
    if false_label_total > 0:
        false_correct_rate = (false_label_judged_true / false_label_total) * 100

    print(f"3️⃣ 原本True被判断错误率:")
    print(f"   标签为True的总次数: {true_label_total}")
    print(f"   错误判断为False的次数: {true_label_judged_false}")
    print(f"   错误率: {true_misjudge_rate:.2f}%")

    print(f"4️⃣ 原本False判断正确率:")
    print(f"   标签为False的总次数: {false_label_total}")
    print(f"   正确判断为True的次数: {false_label_judged_true}")
    print(f"   正确率: {false_correct_rate:.2f}%")

    # 添加标签分布分析
    print(f"\n📊 标签分布:")
    print(f"   True标签占比: {(true_label_total/total_naive_verifications*100):.2f}% ({true_label_total}/{total_naive_verifications})")
    print(f"   False标签占比: {(false_label_total/total_naive_verifications*100):.2f}% ({false_label_total}/{total_naive_verifications})")
    print("-" * 30)
    print()

    # 5. 高级验证准确率（只有完整3次验证且都正确才计入）
    adv_acc = (correct_advanced_count / total_files) * 100 if total_files > 0 else 0
    print(f"5️⃣ 高级验证准确率 (3/3)：{correct_advanced_count}/{total_files} ({adv_acc:.2f}%)")

    # 6. 加减分制评分（新增指标）
    print(f"\n6️⃣ 加减分制评分（对+1分，错-1分）：")
    print(f"   计入评分的文件数：{scored_files}/{total_files}")
    print(f"   总分：{total_score}")
    if scored_files > 0:
        avg_score = total_score / scored_files
        print(f"   平均分：{avg_score:.4f}")
    print("="*50)

    # 输出不完整和成功文件数量信息
    if incomplete_files:
        print(f"\n⚠️  发现 {len(incomplete_files)} 个文件未捕获完整 3 次验证。")
    else:
        print("\n✅ 所有文件均成功捕获 3 次验证结果。")

    if success_files:
        print(f"\n🎉  发现 {len(success_files)} 个文件三次验证均正确（得+1分）。")
    else:
        print("\n⚠️  没有发现三次验证均正确的题目。")

    if failed_files:
        print(f"\n❌  发现 {len(failed_files)} 个文件有完整验证但存在错误（得-1分）。")

    # 如果提供了数据集路径，尝试从数据集中提取题目信息
    dataset_questions = {}
    if dataset_path and os.path.exists(dataset_path):
        print(f"\n📂 从数据集加载题目信息: {dataset_path}")
        dataset_questions = load_questions_from_dataset(dataset_path)
        print(f"   已加载 {len(dataset_questions)} 个题目信息")
    elif dataset_path and not os.path.exists(dataset_path):
        print(f"\n⚠️  数据集文件不存在: {dataset_path}")
    # 导出不完整的题目到 JSONL 文件
    if output_jsonl_path and incomplete_data:
        export_incomplete_to_jsonl(incomplete_data, dataset_questions, output_jsonl_path)

def load_questions_from_dataset(dataset_path):
    """从数据集中加载题目信息"""
    questions = {}

    # 支持多种文件格式
    if dataset_path.endswith('.jsonl'):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            # 假设数据集中有 'idx'、'id' 或 'problem_id' 字段
                            question_id = str(data.get('idx') or data.get('id') or data.get('problem_id') or data.get('ID'))
                            if question_id:
                                questions[question_id] = data
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"❌ 读取JSONL数据集时出错: {e}")

    elif dataset_path.endswith('.json'):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 根据数据结构处理
                if isinstance(data, list):
                    for item in data:
                        question_id = str(item.get('idx') or item.get('id') or item.get('problem_id') or item.get('ID'))
                        if question_id:
                            questions[question_id] = item
                elif isinstance(data, dict):
                    # 可能是字典，键是题目ID
                    for question_id, item in data.items():
                        questions[str(question_id)] = item
        except Exception as e:
            print(f"❌ 读取JSON数据集时出错: {e}")

    else:
        print(f"⚠️  不支持的格式文件: {dataset_path}")

    return questions

def export_incomplete_to_jsonl(incomplete_data, dataset_questions, output_path):
    """将不完整的题目导出到 JSONL 文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_exported = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in incomplete_data:
            question_id = item["id"]

            # 如果数据集中有匹配的题目，直接复制整条原数据
            if question_id in dataset_questions:
                # 直接写入数据集中的原始JSON数据
                dataset_item = dataset_questions[question_id]
                f.write(json.dumps(dataset_item, ensure_ascii=False) + '\n')
                total_exported += 1

    print(f"\n✅ 已将 {total_exported} 个不完整题目导出到: {output_path}")
    print(f"   输出格式: JSONL (与原数据集格式完全一致)")
    print(f"   说明: 直接复制原数据集中idx匹配的整行数据")


def test_scoring_logic():
    """测试新的加减分制评分逻辑"""
    print("🧪 测试加减分制评分逻辑")
    print("-" * 40)

    # 模拟测试数据
    test_cases = [
        {"id": "test1", "judges": [("True", "true"), ("True", "true"), ("True", "true")], "expected_score": 1},
        {"id": "test2", "judges": [("True", "true"), ("True", "false"), ("True", "true")], "expected_score": -1},
        {"id": "test3", "judges": [("True", "false"), ("True", "false"), ("True", "false")], "expected_score": -1},
        {"id": "test4", "judges": [("True", "true"), ("True", "true")], "expected_score": 0},  # 不完整验证，不计分
    ]

    total_score = 0
    scored_files = 0

    for case in test_cases:
        judges = case["judges"]
        file_correct_count = 0

        for label, answer in judges:
            if label.strip().lower() == answer.strip().lower():
                file_correct_count += 1

        # 所有文件都计入评分分母
        scored_files += 1
        if len(judges) == 3:
            if file_correct_count == 3:
                total_score += 1
            else:
                total_score -= 1
        else:
            # 不完整验证文件得0分
            total_score += 0

        print(f"📝 测试 {case['id']}: {len(judges)}次验证，{file_correct_count}次正确")
        if len(judges) == 3:
            print(f"    预期得分: {case['expected_score']}")
        else:
            print(f"    验证不完整，得0分")

    print("-" * 40)
    print(f"📊 测试结果:")
    print(f"    计入评分的文件数: {scored_files}")
    print(f"    总分: {total_score}")
    if scored_files > 0:
        avg_score = total_score / scored_files
        print(f"    平均分: {avg_score:.4f}")

    # 验证逻辑
    expected_total_score = 1 - 1 - 1 + 0  # test1(+1), test2(-1), test3(-1), test4(0) = -1
    expected_scored_files = 4  # 所有测试用例都计入分母
    if total_score == expected_total_score and scored_files == expected_scored_files:
        print("✅ 加减分制评分逻辑测试通过!")
    else:
        print(f"❌ 测试失败: 预期总分{expected_total_score}, 实际总分{total_score}")
        print(f"   预期计入文件数: {expected_scored_files}, 实际计入文件数: {scored_files}")


def main():
    # 硬编码的路径配置 - 使用实际存在的路径进行测试
    # FOLDER_PATH = "./outputs/deepseek-reasoner/judge-cot/SVBench/final"  # 原路径
    FOLDER_PATH = "./outputs/mmlu-cot/deepseek-chat/judge-cot/SVBench/final"  # 实际存在的路径
    # DATASET_PATH = "./data/SVBench/svbench-mmlu_idx.jsonl"    # 数据集路径（正常情况）
    DATASET_PATH = None  # 测试没有数据集的情况
    OUTPUT_PATH = "./test_incomplete_questions.jsonl"                 # 输出文件路径

    print(f"🔧 使用硬编码配置:")
    print(f"   评测文件夹: {FOLDER_PATH}")
    print(f"   数据集: {DATASET_PATH or '不提供数据集'}")
    print(f"   输出文件: {OUTPUT_PATH if DATASET_PATH else '不导出（无数据集）'}")

    # 检查文件夹是否存在
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ 错误: 文件夹 '{FOLDER_PATH}' 不存在")
        # 运行测试逻辑
        test_scoring_logic()
        return

    # 运行实际的准确率计算
    calculate_accuracies(FOLDER_PATH, DATASET_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main()