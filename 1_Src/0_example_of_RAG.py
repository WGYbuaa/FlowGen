from imports import qianfan


if __name__ == "__main__":
    print("nice to meet you!")

    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    speed = "ERNIE-Speed-8K"
    turbo = "ERNIE-4.0-Turbo-8K"
    ernie35 = "ERNIE-3.5-8K-0701"  # 换用3.5试试看

    str1 = "give me some tips to improve my English"

    # prompt
    resp = chat_comp.do(model=ernie35, messages=[{
        "role": "user",
        "content": str1
    }], disable_search=True)

    exist = resp["body"]["result"]

    print(exist)
    print("1!")
    print("2!")
    print("3!")
