from io import BytesIO
from PIL import Image

class preprocessing: 
    def __init__(self) -> None:
           pass
    
    @staticmethod
    def convert_PIL_to_bytes(pil_image: Image) -> BytesIO: 
        """
        Converts a PIL image object to bytes using the specified image format.

        Args:
            pil_image (Image): The PIL image object to convert.

        Returns:
            BytesIO: The converted image bytes.

        """
        with BytesIO() as byte_stream:
            pil_image.save(byte_stream, format='PNG')  # You can specify the format based on your image type, e.g., 'JPEG', 'PNG', etc.
            image_bytes = byte_stream.getvalue()
        return image_bytes