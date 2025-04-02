import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def create_segmentation_masks(annotation_dir, image_dir, output_mask_dir, mask_suffix="_mask", mask_value=255, image_format=".jpg", mask_format=".png"):
    """
    Tạo segmentation masks từ các file annotation XML (PASCAL VOC) cho dataset BCCD.

    Args:
        annotation_dir (str): Đường dẫn đến thư mục chứa các file XML annotation.
        image_dir (str): Đường dẫn đến thư mục chứa các ảnh gốc.
        output_mask_dir (str): Đường dẫn đến thư mục để lưu trữ các ảnh mask đã tạo.
        mask_suffix (str): Hậu tố để thêm vào tên file mask.
        mask_value (int): Giá trị pixel cho các vùng tế bào máu trong mask (thường là 255 cho binary mask).
        image_format (str): Định dạng của ảnh gốc.
        mask_format (str): Định dạng của ảnh mask đầu ra.
    """
    os.makedirs(output_mask_dir, exist_ok=True)

    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Lấy tên file ảnh gốc từ XML
        image_name_element = root.find("filename")
        if image_name_element is not None:
            image_name = image_name_element.text
            # Loại bỏ phần mở rộng nếu có trong tên file XML
            if image_name.endswith(".jpg"):
                image_name = image_name[:-4]
        else:
            print(f"Warning: Không tìm thấy 'filename' trong {xml_file}. Bỏ qua.")
            continue

        # Tạo đường dẫn đầy đủ đến ảnh gốc
        image_path = os.path.join(image_dir, image_name + image_format)

        try:
            image = Image.open(image_path)
            width, height = image.size
        except FileNotFoundError:
            print(f"Warning: Không tìm thấy ảnh {image_path}. Bỏ qua.")
            continue

        # Tạo ảnh mask đen trắng
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Vẽ bounding box lên mask
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Vẽ một hình chữ nhật trắng lên mask tại vị trí bounding box
                draw.rectangle([(xmin, ymin), (xmax, ymax)], fill=mask_value)

        # Lưu ảnh mask
        mask_name = os.path.splitext(xml_file)[0] + mask_suffix + mask_format
        mask_path = os.path.join(output_mask_dir, mask_name)
        mask.save(mask_path)

        print(f"Đã tạo mask cho {image_name}")

if __name__ == "__main__":
    annotation_dir = r"D:\NCKH\BCCD_Dataset\BCCD\Annotations"  # Thay đổi đường dẫn nếu cần
    image_dir = r"D:\NCKH\BCCD_Dataset\BCCD\JPEGImages"     # Thay đổi đường dẫn nếu cần
    output_mask_dir = r"D:\NCKH\BCCD_Dataset\masks"        # Thay đổi đường dẫn nếu cần

    create_segmentation_masks(annotation_dir, image_dir, output_mask_dir)
    print("Hoàn thành việc tạo mask.")