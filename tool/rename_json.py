import os
import json


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        name = name.replace(char, '_')
    return name


def rename_json_files(directory):
    """重命名目录中的所有JSON文件，遇到冲突则跳过"""
    processed = 0
    skipped = 0
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(directory, filename)
        try:
            # 读取JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 获取tool_name字段
            tool_name = data.get('tool_name')
            if not tool_name:
                print(f"⚠️ 跳过: {filename} (缺少tool_name字段)")
                skipped += 1
                continue
            # 清理文件名并添加扩展名
            clean_name = sanitize_filename(tool_name)
            new_name = f"{clean_name}.json"
            new_path = os.path.join(directory, new_name)
            # 如果文件名已经是目标名称，跳过
            if filename == new_name:
                print(f"✅ 已是正确名称: {filename}")
                processed += 1
                continue
            # 检查目标文件名是否已存在
            if os.path.exists(new_path):
                print(f"⚠️ 跳过: {filename} (目标文件 {new_name} 已存在)")
                skipped += 1
                continue
            # 重命名文件
            os.rename(filepath, new_path)
            print(f"🔄 重命名: {filename} → {new_name}")
            processed += 1
        except Exception as e:
            print(f"❌ 错误处理 {filename}: {str(e)}")
    # 输出统计信息
    print("\n=== 处理结果 ===")
    print(f"✅ 成功处理: {processed} 个文件")
    print(f"⚠️ 跳过: {skipped} 个文件")


# 使用示例
if __name__ == "__main__":
    target_directory = input("请输入包含JSON文件的目录路径: ").strip()
    if os.path.isdir(target_directory):
        print(f"\n开始处理目录: {target_directory}\n")
        rename_json_files(target_directory)
        print("\n处理完成！")
    else:
        print("错误: 指定的目录不存在")