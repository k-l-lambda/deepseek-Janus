from openai import OpenAI
import base64
from PIL import Image
import io



def image_to_data_url(image_path):
	with Image.open(image_path) as img:
		buffered = io.BytesIO()
		img.save(buffered, format="PNG")
		img_str = base64.b64encode(buffered.getvalue()).decode()
		return f"data:image/png;base64,{img_str}"


def main ():
	# Load and convert image
	image_url = image_to_data_url("./images/equation.png")

	client = OpenAI(
		api_key="dummy",
		base_url="http://localhost:8000/v1"
	)

	# Vision API
	response = client.chat.completions.create(
		model="janus-1.3b",
		messages=[
			{
				"role": "user",
				"content": [
					{"type": "image_url", "image_url": image_url},
					{"type": "text", "text": "What's in this image?"}
				]
			}
		]
	)

	print(response)

'''# Image generation
response = client.images.generate(
	model="janus-1.3b",
	prompt="A beautiful sunset over mountains",
	n=1,
	size="1024x1024"
)'''


if __name__ == "__main__":
	main()
