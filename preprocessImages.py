
import os
import io
import typer
import datasets
from typing_extensions import Annotated
import zipfile
import PIL.Image
import torch
from tqdm import tqdm

from imageProc import loadJanus, encodeImage



app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main (index_path: Annotated[str, 'index csv file path'], image_dir: Annotated[str, 'source image directory']):
	image_processor, vl_gpt = loadJanus('deepseek-ai/Janus-Pro-7B')

	# load csv dataset from index_path by datasets library
	index_set = datasets.load_dataset('csv', data_files=index_path, split='train')

	target_path = index_path.replace('.csv', '.zip')
	with zipfile.ZipFile(target_path, mode="w", compresslevel=5) as package:
		for row in tqdm(index_set):
			img = PIL.Image.open(os.path.join(image_dir, row['image']))
			emb = encodeImage(img, image_processor, vl_gpt)
			buffer = io.BytesIO()
			torch.save(emb, buffer)
			package.writestr(f"{row['index']}.pt", buffer.getvalue())

	print('Done.')


if __name__ == '__main__':
	app()
