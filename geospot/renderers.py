"""VLM Renderers for Geospot."""

import io
import logging
import urllib.request
from enum import StrEnum
from typing import Any, NotRequired, Optional, TypedDict, Literal, Protocol, cast

import tinker
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Type aliases (actual types come from transformers at runtime)
Tokenizer = Any
ImageProcessor = Any


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    image: str | Image.Image


ContentPart = TextPart | ImagePart
Role = str
Content = str | list[ContentPart]


class Message(TypedDict):
    role: Role
    content: Content
    trainable: NotRequired[bool]


class RenderedMessage(TypedDict):
    prefix: NotRequired[tinker.EncodedTextChunk]
    content: list[tinker.ModelInputChunk]
    suffix: NotRequired[tinker.EncodedTextChunk]


class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    CUSTOMIZED = "customized"


def ensure_text(content: Content) -> str:
    """Extract text content from a message."""
    if isinstance(content, str):
        return content
    return " ".join(part["text"] for part in content if part["type"] == "text")


def _is_empty_chunk(chunk: tinker.types.ModelInputChunk) -> bool:
    """Check if a chunk has no tokens (would cause Tinker BadRequestError)."""
    if isinstance(chunk, tinker.types.EncodedTextChunk):
        return len(chunk.tokens) == 0
    return False


class Renderer(Protocol):
    """Render messages into model inputs."""

    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def _preprocess_message_parts(self, message: Message) -> list[ImagePart | TextPart]:
        return (
            message["content"]
            if isinstance(message["content"], list)
            else [TextPart(type="text", text=message["content"])]
        )

    @property
    def _bos_tokens(self) -> list[int]:
        return []

    def get_stop_sequences(self) -> list[str] | list[int]:
        raise NotImplementedError

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        raise NotImplementedError

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        raise NotImplementedError

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        chunks: list[tinker.types.ModelInputChunk] = []
        if self._bos_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=self._bos_tokens))
        for idx, message in enumerate(messages):
            rendered = self.render_message(idx, message)
            if ob := rendered.get("prefix"):
                if not _is_empty_chunk(ob):
                    chunks.append(ob)
            chunks.extend([x for x in rendered["content"] if x and not _is_empty_chunk(x)])
        new_msg = Message(role=role, content="")
        rendered = self.render_message(len(messages), new_msg)
        if ob := rendered.get("prefix"):
            if not _is_empty_chunk(ob):
                chunks.append(ob)
        if prefill:
            tokens = self.tokenizer.encode(prefill, add_special_tokens=False)
            if tokens:
                chunks.append(tinker.types.EncodedTextChunk(tokens=tokens))
        return tinker.ModelInput(chunks=chunks)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        chunks_weights: list[tuple[tinker.types.ModelInputChunk, float]] = []
        if self._bos_tokens:
            chunks_weights.append(
                (tinker.types.EncodedTextChunk(tokens=self._bos_tokens), 0.0)
            )

        for idx, message in enumerate(messages):
            is_last = idx == len(messages) - 1
            is_assistant = message["role"] == "assistant"
            rendered = self.render_message(idx, message, is_last=is_last)

            ob_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
            if ob := rendered.get("prefix"):
                if not _is_empty_chunk(ob):
                    chunks_weights.append((ob, ob_weight))

            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    has_weight = is_last and is_assistant
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    has_weight = is_assistant
                case TrainOnWhat.ALL_MESSAGES | TrainOnWhat.ALL_TOKENS:
                    has_weight = True
                case TrainOnWhat.CUSTOMIZED:
                    has_weight = message.get("trainable", False)
                case _:
                    raise ValueError(f"Unknown train_on_what: {train_on_what}")

            for part in rendered.get("content", []):
                if part and not _is_empty_chunk(part):
                    chunks_weights.append((part, int(has_weight)))

            if is_last and (tail := rendered.get("suffix")):
                chunks_weights.append((tail, int(has_weight)))

        weights = [w for chunk, w in chunks_weights for _ in range(chunk.length)]
        return (
            tinker.ModelInput(chunks=[c for c, _ in chunks_weights]),
            torch.tensor(weights),
        )


def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, bool]:
    count = response.count(stop_token)
    if count == 0:
        return Message(role="assistant", content=tokenizer.decode(response)), False
    elif count == 1:
        text = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=text), True
    else:
        raise ValueError(f"Expected 1 stop token, got {count}")


class ImageProcessorProtocol(Protocol):
    merge_size: int
    patch_size: int

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: Optional[dict] = None
    ) -> int:
        raise NotImplementedError


def image_to_chunk(
    image_or_str: Image.Image | str, image_processor: ImageProcessorProtocol
) -> tinker.types.ImageChunk:
    if isinstance(image_or_str, str):
        with urllib.request.urlopen(image_or_str) as resp:
            pil_image = Image.open(io.BytesIO(resp.read()))
    else:
        pil_image = image_or_str

    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")

    # Calculate expected tokens for Qwen3-VL
    w, h = pil_image.size
    patches = image_processor.get_number_of_image_patches(h, w, images_kwargs={})
    expected_tokens = patches // (image_processor.merge_size ** 2)

    return tinker.types.ImageChunk(data=buf.getvalue(), format="jpeg", expected_tokens=expected_tokens)


class Qwen3Renderer(Renderer):
    """Qwen3 chat format."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert isinstance(message["content"], str)
        nl = "\n" if idx > 0 else ""
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(f"{nl}<|im_start|>{message['role']}\n", add_special_tokens=False)
        )
        content = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(message["content"] + "<|im_end|>", add_special_tokens=False)
        )
        return RenderedMessage(prefix=prefix, content=[content])

    @property
    def _end_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_token)


class Qwen3VLRenderer(Qwen3Renderer):
    """Qwen3 VL format with image support."""

    image_processor: ImageProcessor

    def __init__(self, tokenizer: Tokenizer, image_processor: ImageProcessor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def _preprocess_message_parts(self, message: Message) -> list[ImagePart | TextPart]:
        chunks: list[ImagePart | TextPart] = []
        for part in super()._preprocess_message_parts(message):
            if part["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_start|>"))
            chunks.append(part)
            if part["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_end|>"))
        return chunks

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        nl = "\n" if idx > 0 else ""
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(f"{nl}<|im_start|>{message['role']}\n", add_special_tokens=False)
        )

        parts = self._preprocess_message_parts(message) + [TextPart(type="text", text="<|im_end|>")]
        content: list[tinker.ModelInputChunk] = [
            image_to_chunk(p["image"], cast(ImageProcessorProtocol, self.image_processor))
            if p["type"] == "image"
            else tinker.EncodedTextChunk(tokens=self.tokenizer.encode(p["text"], add_special_tokens=False))
            for p in parts
        ]
        return RenderedMessage(prefix=prefix, content=content)


def get_renderer(
    name: str, tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    if name == "qwen3":
        return Qwen3Renderer(tokenizer)
    elif name == "qwen3_vl":
        assert image_processor, "qwen3_vl requires image_processor"
        return Qwen3VLRenderer(tokenizer, image_processor)
    raise ValueError(f"Unknown renderer: {name}")
