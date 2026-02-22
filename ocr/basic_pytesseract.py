import asyncio
import pytesseract
from io import BytesIO
from PIL import Image
from fastapi.concurrency import run_in_threadpool
from typing import List

def ocr_image(image_bytes: bytes) -> str:
    """
    Perform OCR on the given image bytes using pytesseract.

    Args:
        image_bytes: The raw bytes of the image.

    Returns:
        str: The extracted text from the image.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang='ara')
        return text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

async def ocr_image_async(image_bytes: bytes) -> str:
    """
    Asynchronous version of the OCR function.

    Args:
        image_bytes: The raw bytes of the image.

    Returns:
        str: The extracted text from the image.
    """
    return await run_in_threadpool(ocr_image, image_bytes)

async def ocr_images_batch(image_bytes_list: List[bytes]) -> List[str]:
    """
    Process a list of images in parallel using asyncio.gather.

    Args:
        image_bytes_list: A list of raw image bytes.

    Returns:
        List[str]: A list of extracted texts.
    """
    tasks = [ocr_image_async(image_bytes) for image_bytes in image_bytes_list]
    return await asyncio.gather(*tasks)
