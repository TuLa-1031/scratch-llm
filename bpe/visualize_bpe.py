
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Load tokenizer
# ============================================================

TOKENIZER_PATH = "output/tokenizer.pkl"
OUTPUT_DIR = Path("output/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(TOKENIZER_PATH, "rb") as f:
    vocab, merged_list = pickle.load(f)

print(f"Vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merged_list)}")


def decode_token(token: bytes) -> str:
    return token.decode("utf-8", errors="replace")


# ============================================================
# 2. Semantic embedding map
# ============================================================

def visualize_semantics(
    vocab: dict[int, bytes],
    max_tokens: int = 500,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Chọn các token giống từ hoàn chỉnh để giảm nhiễu.
    Encode token bằng pretrained sentence embedding model,
    sau đó giảm chiều bằng PCA.
    """

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    special_tokens = {
        token for token in vocab.values()
        if token.startswith(b"<|") and token.endswith(b"|>")
    }

    candidates = []

    for token_id, token_bytes in vocab.items():
        if token_bytes in special_tokens:
            continue

        text = decode_token(token_bytes)

        # Bỏ token rỗng, token có ký tự thay thế UTF-8,
        # và token chỉ chứa whitespace.
        if not text or "\ufffd" in text or text.isspace():
            continue

        # Lấy token chứa chữ cái, có độ dài vừa phải.
        if not any(ch.isalpha() for ch in text):
            continue

        if len(text) < 3 or len(text) > 30:
            continue

        # Không đưa chuỗi có quá nhiều ký tự điều khiển vào model.
        if any(ord(ch) < 32 for ch in text):
            continue

        candidates.append((token_id, text))

    # Bỏ trùng nhãn và giới hạn kích thước để dễ visualize.
    seen = set()
    selected = []

    for token_id, text in candidates:
        normalized = text.strip().lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        selected.append((token_id, text))

    if len(selected) < 3:
        print("Không đủ token phù hợp để tạo semantic map.")
        return

    # Chọn mẫu phân bố đều theo ID để không chỉ lấy token đầu tiên.
    if len(selected) > max_tokens:
        indices = np.linspace(
            0, len(selected) - 1, max_tokens, dtype=int
        )
        selected = [selected[i] for i in indices]

    ids = [x[0] for x in selected]
    labels = [x[1] for x in selected]

    print(f"Embedding {len(labels)} tokens...")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        labels,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # PCA không cần UMAP, phù hợp để chạy thử nhanh.
    coords = PCA(n_components=2).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.scatter(coords[:, 0], coords[:, 1], s=16, alpha=0.65)

    # Chỉ ghi nhãn một tập con để tránh chồng chữ.
    label_count = min(120, len(labels))
    label_indices = np.linspace(
        0, len(labels) - 1, label_count, dtype=int
    )

    for i in label_indices:
        ax.annotate(
            labels[i],
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.85,
        )

    ax.set_title("BPE Vocabulary: Semantic Embedding Map (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "semantic_map.png"
    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved: {output_path}")


# ============================================================
# 3. BPE merge tree
# ============================================================

def visualize_merge_tree(
    merged_list: list[tuple[bytes, bytes]],
    vocab: dict[int, bytes],
    merge_index: int = 100,
):
    """
    Tái dựng cây cho một token được tạo ở merge_index.

    Mỗi token merge được tạo từ hai token có sẵn trước đó.
    merge_index dùng chỉ số bắt đầu từ 0.
    """

    if not merged_list:
        print("Không có merge rules.")
        return

    merge_index = max(0, min(merge_index, len(merged_list) - 1))

    # Map token bytes -> node ID trong cây.
    token_to_node = {}
    nodes = {}
    next_node_id = 0

    # Khởi tạo 256 byte cơ bản.
    for i in range(256):
        token = bytes([i])
        token_to_node[token] = next_node_id
        nodes[next_node_id] = {
            "label": repr(decode_token(token)),
            "children": [],
        }
        next_node_id += 1

    # Thêm special tokens để tránh nhầm ID với byte cơ bản.
    for token in vocab.values():
        if token not in token_to_node and (
            token.startswith(b"<|") and token.endswith(b"|>")
        ):
            token_to_node[token] = next_node_id
            nodes[next_node_id] = {
                "label": decode_token(token),
                "children": [],
            }
            next_node_id += 1

    merge_roots = []

    for i, (left, right) in enumerate(merged_list[:merge_index + 1]):
        # Nếu token con chưa được tạo, tree không thể được tái dựng
        # theo thứ tự merge; báo lỗi thay vì tạo cây sai.
        if left not in token_to_node or right not in token_to_node:
            raise ValueError(
                f"Merge {i}: token con chưa tồn tại trong các merge trước. "
                "Kiểm tra thứ tự merge hoặc format tokenizer."
            )

        merged = left + right
        left_node = token_to_node[left]
        right_node = token_to_node[right]

        node_id = next_node_id
        next_node_id += 1

        nodes[node_id] = {
            "label": decode_token(merged),
            "children": [left_node, right_node],
        }

        token_to_node[merged] = node_id
        merge_roots.append(node_id)

    root = merge_roots[-1]

    # Tính tọa độ cây theo thứ tự duyệt.
    positions = {}
    leaf_counter = [0]

    def layout(node_id, depth=0):
        children = nodes[node_id]["children"]

        if not children:
            x = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            child_x = [
                layout(child, depth + 1)
                for child in children
            ]
            x = sum(child_x) / len(child_x)

        positions[node_id] = (x, -depth)
        return x

    layout(root)

    fig_width = max(12, min(28, leaf_counter[0] * 0.7))
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    def draw(node_id):
        x, y = positions[node_id]
        children = nodes[node_id]["children"]

        for child in children:
            cx, cy = positions[child]
            ax.plot([x, cx], [y, cy], linewidth=1.2)
            draw(child)

        label = nodes[node_id]["label"]
        ax.scatter([x], [y], s=35, zorder=3)
        ax.annotate(
            label[:24],
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    draw(root)

    ax.set_title(
        f"BPE Merge Tree — merge #{merge_index}: "
        f"{decode_token(merged_list[merge_index][0] + merged_list[merge_index][1])!r}"
    )
    ax.axis("off")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"merge_tree_{merge_index}.png"
    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved: {output_path}")


# ============================================================
# 4. Run
# ============================================================

if __name__ == "__main__":
    visualize_semantics(vocab, max_tokens=500)

    visualize_merge_tree(
        merged_list,
        vocab,
        merge_index=min(100, len(merged_list) - 1),
    )
