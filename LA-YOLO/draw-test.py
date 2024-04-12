import os
import cv2
from tqdm import tqdm
#修改map_out文件路径即可
#如需要在图中打印label和置信度信息 解除相应cv2.putText行注释
mapout_file="/home/aquar/lsj/yolostudy/result/map_out"#map_out文件路径

def replace_strings_in_files(directory, output_directory, replacements):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for file_name in tqdm(os.listdir(directory)):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)

            with open(file_path, 'r') as f:
                content = f.read()

            for old_str, new_num in replacements.items():
                content = content.replace(old_str, str(new_num))

            new_file_path = os.path.join(output_directory, file_name)
            with open(new_file_path, 'w') as f:
                f.write(content)

            #print(f"Processed {file_name}. Result saved as {new_file_path}")
def draw_bounding_boxes(image_path, model_boxes, true_boxes, class_labels, output_directory):
    image = cv2.imread(image_path)


    for box,label in model_boxes:
        x,y,w,h = box
        cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 1)  # 绘制模型标注的锚框，红色
        #cv2.putText(image, class_labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)  # 绘制标签，红色

    #for box, label in true_boxes:
    #    x,y,w,h  = box

    #    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 1)  # 绘制正确锚框，绿色
    #    cv2.putText(image, class_labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)  # 绘制标签，绿色

    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_directory, image_name)
    cv2.imwrite(output_path, image)

    #print(f"Processed {image_path}. Result saved as {output_path}")

def draw_together_boxes(image_path, model_boxes, true_boxes, class_labels, output_directory):
    image = cv2.imread(image_path)


    for box,label in model_boxes:
        x,y,w,h = box
        cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 1)  # 绘制模型标注的锚框，红色
        #cv2.putText(image, class_labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)  # 绘制标签，红色

    for box, label in true_boxes:
        x,y,w,h  = box

        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 1)  # 绘制正确锚框，绿色
        #cv2.putText(image, class_labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)  # 绘制标签，绿色

    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_directory, image_name)
    cv2.imwrite(output_path, image)

    #print(f"Processed {image_path}. Result saved as {output_path}")
# 示例用法
replacements = {
    'flying object': 0,  # 将 'string1' 替换为 10
    'vehicle': 1,  # 将 'string2' 替换为 20
    'watercraft': 2   # 将 'string3' 替换为 30
}

directory =   mapout_file+'/detection-results'
output_directory = mapout_file+'/detection-results-i'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
replace_strings_in_files(directory, output_directory, replacements)
directory =   mapout_file+'/ground-truth'
output_directory = mapout_file+'/ground-truth-i'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
replace_strings_in_files(directory, output_directory, replacements)


image_directory = mapout_file+'/images-optional'  
model_boxes_directory = mapout_file+'/detection-results-i'  
true_boxes_directory = mapout_file+'/ground-truth-i'  
output_directory = mapout_file+'/draw_wrong'  
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
output_directory_draw_all = mapout_file+'/draw_together' 
if not os.path.exists(output_directory_draw_all):
    os.makedirs(output_directory_draw_all)
class_labels = ['flying object', 'vehicle', 'watercraft']  # 替换为实际的类别列表


# 遍历图片目录下的所有图片
status=mapout_file+'/results/status'
desired_states = ['REPEATED MATCH!', 'INSUFFICIENT OVERLAP']
wrongs=[]
print("获取图像检测异常文件……")
for statu in tqdm(os.listdir(status)):
    statu_path = os.path.join(status, statu)
    with open(statu_path, 'r') as f:
        content = f.read()
        if any(state in content for state in desired_states):
            wrongs.append(statu.split('.')[0].split('/')[0])
print("共"+str(len(wrongs))+"个图像检测异常文件")
with open(status+'/wrongs.txt', 'w') as f:
    for wrong in wrongs:
        f.write(wrong+'\n')
print("start drawing")
for image_name in tqdm(os.listdir(image_directory)):
    if image_name.endswith('.jpg') and image_name.split('.')[0].split('/')[0] in wrongs:
        image_path = os.path.join(image_directory, image_name)
        # 读取模型标注的锚框文件
        model_boxes_path = os.path.join(model_boxes_directory, image_name.replace('.jpg', '.txt'))
        with open(model_boxes_path, 'r') as f:
            #model_boxes = [list(map(int, line.strip().split(' ')[2:])) for line in f]
            model_boxes= [([int(coord) for coord in line.strip().split(' ')[2:]], int(line.strip().split(' ')[0])) for line in f]
        # 读取正确锚框文件
        true_boxes_path = os.path.join(true_boxes_directory, image_name.replace('.jpg', '.txt'))
        with open(true_boxes_path, 'r') as f:
            true_boxes = [([int(coord) for coord in line.strip().split(' ')[1:]], int(line.strip().split(' ')[0])) for line in f]

        # 绘制并保存结果图像
        draw_bounding_boxes(image_path, model_boxes, true_boxes, class_labels, output_directory)
        draw_together_boxes(image_path, model_boxes, true_boxes, class_labels, output_directory_draw_all)
print("completed")
print("异常检测数据可视")
print("模型检测可视结果保存路径:"+output_directory)
print("对比可视结果保存路径:"+output_directory_draw_all)
