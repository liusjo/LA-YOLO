
import os
#使用说明
#替换下面两个路径，对预测的txt和真实的txt都要换一遍
directory = './result/map_out/detection-results'  # 替换为指定的目录路径
output_directory = 'result/map_out/detection-results-i'  # 替换为保存处理结果的目录路径

def replace_strings_in_files(directory, output_directory, replacements):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(directory):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)

            with open(file_path, 'r') as f:
                content = f.read()

            for old_str, new_num in replacements.items():
                content = content.replace(old_str, str(new_num))

            new_file_path = os.path.join(output_directory, file_name)
            with open(new_file_path, 'w') as f:
                f.write(content)

            print(f"Processed {file_name}. Result saved as {new_file_path}")

# 示例用法
replacements = {
    'flying object': 0,  # 将 'string1' 替换为 10
    'vehicle': 1,  # 将 'string2' 替换为 20
    'watercraft': 2   # 将 'string3' 替换为 30
}

replace_strings_in_files(directory, output_directory, replacements)