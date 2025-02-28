
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import VLChatProcessor



def loadJanus (path, device='cuda'):
	vl_chat_processor = VLChatProcessor.from_pretrained(path)

	config = AutoConfig.from_pretrained(path)
	language_config = config.language_config
	vl_gpt = AutoModelForCausalLM.from_pretrained(path, language_config=language_config, trust_remote_code=True)

	vl_gpt.vision_model.to(torch.bfloat16).to(device)
	vl_gpt.aligner.to(torch.bfloat16).to(device)

	return vl_chat_processor.image_processor, vl_gpt


def encodeImage (image, image_processor, vl_gpt, do_align=False):
	pixels = image_processor([image], return_tensors='pt').pixel_values
	x = vl_gpt.vision_model(pixels.to(torch.bfloat16).cuda())
	if do_align:
		x = vl_gpt.aligner(x)

	return x
