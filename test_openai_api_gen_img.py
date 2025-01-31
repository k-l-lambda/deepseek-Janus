from openai import OpenAI
import base64
from PIL import Image
import io
import os
import time



def image_to_data_url(image_path):
	with Image.open(image_path) as img:
		buffered = io.BytesIO()
		img.save(buffered, format="PNG")
		img_str = base64.b64encode(buffered.getvalue()).decode()
		return f"data:image/png;base64,{img_str}"

def save_generated_images(response_data, output_dir="images"):
	print(f'{type(response_data[0].b64_json)=}')
	# Create output directory if not exists
	os.makedirs(output_dir, exist_ok=True)

	saved_paths = []
	timestamp = int(time.time())

	for i, img_data in enumerate(response_data):
		if "b64_json" in dir(img_data):
			# Decode base64
			image_bytes = base64.b64decode(img_data.b64_json)

			# Create image from bytes
			image = Image.open(io.BytesIO(image_bytes))

			# Save image
			filename = f"{output_dir}/generated_{timestamp}_{i}.local.png"
			image.save(filename)
			saved_paths.append(filename)

	return saved_paths

def main():
	# Load and convert image
	image_url = image_to_data_url("./images/equation.png")

	client = OpenAI(
		api_key="dummy",
		base_url="http://localhost:8000/v1"
	)

	response = client.images.generate(
		model="janus-1.3b",
		prompt="A beautiful sunset over mountains",
		n=1,
		size="384x384",
		response_format="b64_json",
	)

	# Save generated images
	saved_files = save_generated_images(response.data)
	print(f"Images saved to: {saved_files}")



if __name__ == "__main__":
	main()
