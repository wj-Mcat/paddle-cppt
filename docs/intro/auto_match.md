# Auto Match

当生成Diff文件之后，需要对names进行映射，这个工作很无聊枯燥 😒 😡，明明很简单的事情却要我们优秀的开发者来完成，此时我的内心是🙅 🙅 🙅

## 技术原理

* 设置相似关键字词表来guess layer 的类型，比如包含：`embedding`, `embed`等关键字即为Embedding类型的Layer，包含：`linear`, `fc`等关键字的layer name即为
* 在相同类型layer中选择最相似的layer进行映射，比如：torch_model.word_embedding需要。