from io import BytesIO
from PIL import Image
import base64
import re


def get_text_from_class(class_name: str, predict: list) -> str:
    """
    Returns the text associated with a given class name from a list of predictions.

    Args:
        class_name (str): The name of the class to search for.
        predict (list): A list of prediction items, each containing a "cls" field and a "text" field.

    Returns:
        str: The text associated with the given class name, or an empty string if no match is found.
    """
    for item in predict:
        if item["cls"] == class_name:
            return item["text"]
    return ""

def get_text_from_classes(class_name1: str, class_name2: str, predict: list) -> str:
    """
    This function takes in two class names and a list of predictions and returns a string.
    
    Parameters:
        class_name1 (str): The name of the first class.
        class_name2 (str): The name of the second class.
        predict (list): A list of predictions.
        
    Returns:
        str: The concatenated text from the two classes if both are non-empty. Otherwise, it returns the text from the first class.
    """
    text1 = get_text_from_class(class_name1, predict)
    text2 = get_text_from_class(class_name2, predict)
    if text1 and text2:
        return text1 + " " + text2
    else:
        return text1
    

def get_face_bbox_from_output(predict: list) -> list:
    """
    Get the bounding box of the face from the given prediction.

    Parameters:
    - predict (list): A list of dictionaries representing the prediction output.

    Returns:
    - str: The bounding box coordinates of the face as a string.
    """
    for item in predict:
        if item["cls"] == "face":
            return item["bbox"]
        
async def get_base64_face_image(image_contents: bytes, predict: list) -> str:
    """
    Generate the base64 representation of a cropped face image.

    Args:
        image_contents (bytes): The contents of the original image.
        predict (list): The predicted bounding box coordinates of the face.

    Returns:
        str: The base64 representation of the cropped face image.

    Raises:
        None
    """
    bbox = get_face_bbox_from_output(predict)
    with BytesIO(image_contents) as input_stream:
        image = Image.open(input_stream)
        crop_image = image.crop(bbox)
        
        with BytesIO() as output_stream:
            crop_image.save(output_stream, format="PNG")
            # Encode the byte stream as base64 and return
            return base64.b64encode(output_stream.getvalue()).decode("utf-8")


def convert_date_format(input_string: str) -> str:
    """
    Convert the date format from a string in Vietnamese to a standard format.

    Args:
        input_string (str): The input string in Vietnamese.

    Returns:
        str: The formatted date in the format "dd/mm/yyyy" if a match is found, otherwise None.
    """
    pre_processing = ' '.join(input_string.replace(".", " ").split())

    # Search for the day, month, and year in the input string
    match = re.search(r'Ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})', pre_processing)

    # If a match is found, format the date
    if match:
        day, month, year = match.groups()
        return f"{day}/{month}/{year}"
    else:
        return None