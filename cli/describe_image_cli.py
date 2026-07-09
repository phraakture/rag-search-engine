import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a text query using an image for better search results"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--query", type=str, required=True, help="Text query to rewrite"
    )
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
