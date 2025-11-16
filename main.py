# import sys
# import os
#
# if __name__ == "__main__":
#     cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
#
#     if cmd == "train":
#         os.system("python train.py --data data/sample.txt --out ckpt.pt --epochs 10 --batch_size 1 --block_size 32 "
#                   "--lr 3e-4 --device mps")
#     elif cmd == "generate":
#         os.system('python generate.py --ckpt ckpt.pt --prompt "Hey there! " --max_new_tokens 500 --device mps')
#     elif cmd == "chat":
#         os.system("python chat.py --ckpt ckpt.pt --device mps")


# main.py
import os
import sys

if __name__ == "__main__":
    print("\n=== LLM Runner ===")
    print("1) Train the model")
    print("2) Generate text")
    print("3) Chat with the model")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        # Train
        os.system(f"{sys.executable} train.py --data data/sample.txt --out ckpt.pt --epochs 10 --batch_size 8 "
                  f"--block_size 128 --n_layer 6 --n_head 6 --n_embd 384 --lr 3e-4 --device mps --tokenizer bpe "
                  f"--vocab_size 2000")

    elif choice == "2":
        # Generate
        os.system(f'{sys.executable} generate.py --ckpt ckpt.pt --prompt "Hey there! " --max_new_tokens 300 '
                  f'--temperature 0.6 --top_k 40 --device mps')

    elif choice == "3":
        # Chat
        os.system(f"{sys.executable} chat.py --ckpt ckpt.pt --device mps --temperature 0.6 --top_k 40")

    else:
        print("Invalid option. Please run again and choose 1, 2, or 3.")
