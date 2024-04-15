# LLMs踩坑集合--Inference篇



### 1. 动态分配lora adapter

先上错误方案代码，初始化一个base model ，然后通过`load_adapter()，set_adapter()`方式动态加载适配器，模型回答效果错乱。原因在于一旦通过`model.set_adapter("sql_lora")` 设置适配器，原base model的参数会受到影响，但是如果仅调用load_adapter()方法则不会影响。

```python
class Qwen(LLM):
    model_type = "qwen"

    def __init__(self, cfg):
        model_path = cfg.get("model_path")
        # 加载原模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", trust_remote_code=True, bf16=True
        )
        self.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )  # 可指定不同的生成长度、top_p等相关超参
        # 加载预备训练模型
        if cfg.get("lora_path") is not None:
            self.model.load_adapter(cfg["lora_path"], adapter_name="agent_lora"
            )  # 产生一个融合adapter的模型
        if cfg.get("lora_2_path") is not None:
            self.model.load_adapter(
                cfg["lora_2_path"], adapter_name="sql_lora"
            )  # 添加adapter，不改变原model的权重

    def original_generate(self, prompt, history=None):
        with self.model.disable_adapter():
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=history,
                generation_config=self.generation_config,
            )
            return response

    def generate(self, prompt, history=None):
        self.model.set_adapter("agent_lora")
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            generation_config=self.generation_config,
        )
        return response

    def sql_generate(self, prompt, history=None):
        self.model.set_adapter("sql_lora")  # 这个方法会导致模型的方法错乱
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            generation_config=self.generation_config,
        )
        return response
```











