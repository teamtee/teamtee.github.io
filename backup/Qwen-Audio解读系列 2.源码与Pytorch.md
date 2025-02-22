
## Auto类
[参考](https://huggingface.co/docs/transformers/model_doc/auto)
transformers提供了一系列的自动函数类，他们提供了通用的加载方式
- AutoConfig
- AutoTokenizer
- AutoFeatureExtractor
- AutoProcessor
- AutoModel
- AutoModelFor...
	- AutoModelForCausalLM
	- ...
其中值得一提的是
- AutoModel和AutoModelFor...的区别是是否包含输出头，比如最后将输出转化为词表大小来输出
```plaintext
(lm_head): Linear(in_features=3584, out_features=152064, bias=False)
```
- AutoModel和AutoModelFor...的加载模型两种方式
```
model = AutoModel.from_pretrained(checkpoint) # 加载权重
model = AutoModel.from_config(config) # 不加载权重
```
实际上我们可能会采用更加具体的模型，比如`Qwen2AudioForConditionalGeneration`

## 实例模型
### PreTrainedModel

PreTrainedModel 是一个抽象类，定义了必须的操作

```
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
```
 **`nn.Module`**：
    - PyTorch 的基础模块类，为模型提供了基本的神经网络功能，例如参数管理、前向传播等。
**`ModuleUtilsMixin`**：
    - 提供了一些通用的模块工具方法，例如处理输入张量的形状、设备分配等。
	    - save_pretrained
	    - from_pretrained
**`GenerationMixin`**：
    - 提供了文本生成相关的功能，例如 `generate()` 方法，支持多种生成策略（如贪婪解码、Beam Search、随机采样等）。
**`PushToHubMixin`**：
    - 提供了将模型推送到 Hugging Face Hub 的功能，方便用户共享和复用模型。
 **`PeftAdapterMixin`**：
    - 提供了对 PEFT（Parameter-Efficient Fine-Tuning）适配器的支持，允许用户在微调时使用更高效的参数优化方法。

`PreTrainedModel`中，
- `forward`,继承自`nn.Module`必须实现的，通常由具体的子类实现
- `generate`，继承自`GenerationMixin`,通过调用`forward`实现
下面是Qwen2.5的系列模型
### Qwen2实例

- `Qwen2PreTrainedModel(PreTrainedModel)` # 继承自抽象类，实现了基础方法，没有forward方法
- `Qwen2Model(Qwen2PreTrainedModel)` # 没有输出头,有forward方法
- `Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin)` # 有输出头，有forward和generate方法
#### Qwen2PreTrainedModel
#### Qwen2Model
`Qwen2Model`的有`forward`方法，没有`generate`方法
```python
def forward(
	self,
	input_ids: torch.LongTensor = None,
	attention_mask: Optional[torch.Tensor] = None,
	position_ids: Optional[torch.LongTensor] = None,
	past_key_values: Optional[List[torch.FloatTensor]] = None,
	inputs_embeds: Optional[torch.FloatTensor] = None,
	use_cache: Optional[bool] = None,
	output_attentions: Optional[bool] = None,
	output_hidden_states: Optional[bool] = None,
	return_dict: Optional[bool] = None,
	cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
...
return BaseModelOutputWithPast(
		last_hidden_state=hidden_states,
		past_key_values=next_cache,
		hidden_states=all_hidden_states,
		attentions=all_self_attns,

	)
```

训练时关注下面的输出，训练时是并行计算的，只需要进行一次forward
- `input_ids /inputs_embeds`：两者只能选择一个输入，`(batch_size, sequence_length)/(batch_size, sequence_length,embedding_dim)`
- attention_mask：指定掩码矩阵，1表示有效位置，0表示无效位置，`(batch_size, sequence_length)`
注意输出的
- `last_hidden_state`:最后一层输出，`(batch_size, sequence_length,embedding_dim)`

```python
def forward(
	self,
	input_ids: torch.LongTensor = None,
	attention_mask: Optional[torch.Tensor] = None,
	position_ids: Optional[torch.LongTensor] = None,
	past_key_values: Optional[List[torch.FloatTensor]] = None,
	inputs_embeds: Optional[torch.FloatTensor] = None,
	labels: Optional[torch.LongTensor] = None,
	use_cache: Optional[bool] = None,
	output_attentions: Optional[bool] = None,
	output_hidden_states: Optional[bool] = None,
	return_dict: Optional[bool] = None,
	cache_position: Optional[torch.LongTensor] = None,
	num_logits_to_keep: int = 0,
	**loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
···
outputs = self.model(
	input_ids=input_ids,
	attention_mask=attention_mask,
	position_ids=position_ids,
	past_key_values=past_key_values,
	inputs_embeds=inputs_embeds,
	use_cache=use_cache,
	output_attentions=output_attentions,
	output_hidden_states=output_hidden_states,
	return_dict=return_dict,
	cache_position=cache_position,
)
···
hidden_states = outputs[0]
# Only compute necessary logits, and do not upcast them to float if we are not computing the loss
logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
loss = None
if labels is not None:
	loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)
if not return_dict:
	output = (logits,) + outputs[1:]
	return (loss,) + output if loss is not None else output
return CausalLMOutputWithPast(
	loss=loss,
	logits=logits,
	past_key_values=outputs.past_key_values,
	hidden_states=outputs.hidden_states,
	attentions=outputs.attentions,
)
```
#### `Qwen2ForCausalLM`
继承了(Qwen2PreTrainedModel, GenerationMixin)，具备`generate`方法和`forward`方法

###### forward
```python
def forward(
	self,
	input_ids: torch.LongTensor = None,
	attention_mask: Optional[torch.Tensor] = None,
	position_ids: Optional[torch.LongTensor] = None,
	past_key_values: Optional[List[torch.FloatTensor]] = None,
	inputs_embeds: Optional[torch.FloatTensor] = None,
	labels: Optional[torch.LongTensor] = None,
	use_cache: Optional[bool] = None,
	output_attentions: Optional[bool] = None,
	output_hidden_states: Optional[bool] = None,
	return_dict: Optional[bool] = None,
	cache_position: Optional[torch.LongTensor] = None,
	num_logits_to_keep: int = 0,
	**loss_kwargs,
	) -> Union[Tuple, CausalLMOutputWithPast]:
	...
	outputs = self.model(
		input_ids=input_ids,
		attention_mask=attention_mask,
		position_ids=position_ids,
		past_key_values=past_key_values,
		inputs_embeds=inputs_embeds,
		use_cache=use_cache,
		output_attentions=output_attentions,
		output_hidden_states=output_hidden_states,
		return_dict=return_dict,
		cache_position=cache_position,
	)
	hidden_states = outputs[0]
	# Only compute necessary logits, and do not upcast them to float if we are not computing the loss
	logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
	loss = None
	if labels is not None:
		loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)
	if not return_dict:
		output = (logits,) + outputs[1:]
		return (loss,) + output if loss is not None else output
	return CausalLMOutputWithPast(
		loss=loss,
		logits=logits,
		past_key_values=outputs.past_key_values,
		hidden_states=outputs.hidden_states,
		attentions=outputs.attentions,
		)
...
```
训练：
- `input_ids /inputs_embeds`：两者只能选择一个输入，`(batch_size, sequence_length)/(batch_size, sequence_length,embedding_dim)`
- `labels`:必须给定
- `attention_mask`：指定掩码矩阵，1表示有效位置，0表示无效位置，`(batch_size, sequence_length)`,可选

推理：
- `input_ids /inputs_embeds`：两者只能选择一个输入
- `num_logits_to_keep`:会被设置为1
- 使用kvcache时给定下面的值
	- past_key_values
	- use_cache
##### generate

`generate`方法输入输出如下

```python
def generate(
	self,
	inputs: Optional[torch.Tensor] = None,
	generation_config: Optional[GenerationConfig] = None,
	logits_processor: Optional[LogitsProcessorList] = None,
	stopping_criteria: Optional[StoppingCriteriaList] = None,
	prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
	synced_gpus: Optional[bool] = None,
	assistant_model: Optional["PreTrainedModel"] = None,
	streamer: Optional["BaseStreamer"] = None,
	negative_prompt_ids: Optional[torch.Tensor] = None,
	negative_prompt_attention_mask: Optional[torch.Tensor] = None,
	**kwargs,
	) -> Union[GenerateOutput, torch.LongTensor]:
	# 通常只返回generate_id
	return input_ids
	
```

#### `Qwen2AudioForConditionalGeneration`


在transformer库的下面可以看到如下的继承结构
```python
Qwen2AudioPreTrainedModel(PreTrainedModel)
Qwen2AudioForConditionalGeneration(Qwen2AudioPreTrainedModel, GenerationMixin)
	- AutoModel
	- Qwen2AudioMultiModalProjector
	- AutoModelForCausalLM
```
`Qwen2AudioForConditionalGeneration`实现如下
 
```
forward(
		input_ids: torch.LongTensor = None,
		input_features: torch.FloatTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		feature_attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	- ) -> Union[Tuple, Qwen2AudioCausalLMOutputWithPast]
		-> return (
		loss=loss,
		logits=logits,
		past_key_values=outputs.past_key_values,
		hidden_states=outputs.hidden_states,
		attentions=outputs.attentions,
		attention_mask=attention_mask,
		)
```

`Qwen2AudioCausalLMOutputWithPast`可以理解為就是一個有序字典




