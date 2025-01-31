from openai import OpenAI
import base64
from PIL import Image
import io



def main ():
	client = OpenAI(
		api_key="dummy",
		base_url="http://localhost:8000/v1"
	)

	response = client.completions.create(
		model="janus-pro-7b",
		prompt="A beautiful sunset over mountains",
	)

	print(response)


if __name__ == "__main__":
	main()
