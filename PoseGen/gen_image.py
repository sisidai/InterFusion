from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import argparse
import os
import torch
import time


# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
stage_3.enable_model_cpu_offload()


def main(in_path, out_path, num_images_per_text):

    lines = []
    with open(in_path, "r") as file:
        for line in file:
            lines.append(line.strip())

    texts = list(set(lines))
    num_texts = len(texts)
    print(f"[INFO] Total {len(lines)} lines loaded, including {num_texts} unique texts")

    time1 = time.time()
    for i in range(num_texts):
        name = texts[i]
        os.makedirs(out_path, exist_ok=True)

        prompt = f"a photo of a person {name}, full body"
        print(f"[INFO] Generate images from <<< {prompt} >>>")
        # text embeds
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

        time2 = time.time()
        for j in range(num_images_per_text):
            time3 = time.time()

            generator = torch.manual_seed(i)
            # stage 1
            image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
            # stage 2
            image = stage_2(
                image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
            ).images
            # stage 3
            image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
            image[0].save(os.path.join(out_path, f"{name.replace(' ', '_')}_{j}.png"))

            print(f"[INFO] {j+1}/{num_images_per_text} images generated, time used for current image: {time.time() - time3}")
        print(f"[INFO] {i+1}/{num_texts} prompts generated, time used for current prompt: {time.time() - time2}")
    print(f"[INFO] Total time used: {time.time() - time1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_path', type=str, default='prompt.txt')
    parser.add_argument('-out_path', type=str, default='results/images')
    parser.add_argument('-num_images_per_text', type=int, default=3)

    args = parser.parse_args()

    main(args.in_path, args.out_path, args.num_images_per_text)