import torch
import clip


def text_prompt(data, prompt=4):
    # 默认采用第4种aug_prompt
    cap_prompt = [f"{{}}"]

    gen_prompt = [f"a photo of {{}}",
                  f"a picture of action {{}}",
                  f"Playing action of {{}}",
                  f"Playing a kind of action, {{}}",
                  f"Doing a kind of action, {{}}",
                  f"Can you recognize the action of {{}}?",
                  f"Video classification of {{}}",
                  f"A video of {{}}"]

    cus_prompt = [f"a photo of {{}}, a type of surgical action",
                  f"Surgical action of {{}}",
                  f"{{}}, a surgical action",
                  f"{{}}, this is a surgical action",
                  f"{{}}, a video of surgical action",
                  f"Look, the surgeon is {{}}",
                  f"The doctor is performing {{}}",
                  f"The surgeon is performing {{}}"]

    aug_prompt = [f"a photo of {{}}, a type of surgical action",
                f"a picture of action {{}}",
                f"Surgical action of {{}}",
                f"{{}}, an action",
                f"{{}} this is an action",
                f"{{}}, a video of surgical action",
                f"Playing action of {{}}",
                f"{{}}",
                f"Playing a kind of action, {{}}",
                f"Doing a kind of action, {{}}",
                f"Look, the surgeon is {{}}",
                f"Can you recognize the action of {{}}?",
                f"Video classification of {{}}",
                f"A video of {{}}",
                f"The doctor is performing {{}}",
                f"The surgeon is performing {{}}"]

    if prompt == 0:
        text_aug = list(cap_prompt)
    elif prompt == 1:
        text_aug = list(gen_prompt)
    elif prompt == 2:
        text_aug = list(cus_prompt)
    else:
        text_aug = list(aug_prompt)

    text_dict = {}
    num_text_aug = len(text_aug)
    
    '''
    data.classes是
    [
        [0, "dissecting connective tissue"],
        [1, "tying a knot"],
        [2, "driving the needle tip"]
    ]
    '''
    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = torch.cat([v for k, v in text_dict.items()]) # 将text_dict中所有value存储的张量全部连接成一个大张量classes

    # torch.cat()用于连接多个张量
    # clip.tokenize()返回张量
    # text_dict键是一个数字（从0开始），值是张量
    # 例如：text_dict[0] = torch.cat([clip.tokenize("a photo of dissecting connective tissue"), clip.tokenize("a photo of tying a knot"), clip.tokenize("a photo of driving the needle tip")])

    return classes, num_text_aug, text_dict