## Attention is all you need (Transformer)
- RNN 단점
- CNN, RNN 모델 없이 단순히 attention mechanism만으로 모델 구성
- long sequence에 의해 학습 내용이 소실되는 것이 없다.
- 학습시간이 빠르다

### Model Architecture
![Transformer Model Architecture](https://miro.medium.com/max/642/1*1BFAQXkNiLySIhB__24EkQ.png)

## 기본적인 과제
- input sequence는 어떻게 생겼는가?
- sequence 가 가지는 성질을 어떻게 해결하는가?
- self attention 이라는 것은 무엇인가?
- layer normalization은 무엇인가?
- self attention vs masked self attention 의 차이는?
- encoder - decoder attention은 무엇인가?

#### Transformer Black Box
![seq2seq model](https://miro.medium.com/max/700/1*jKjKXD0zqUTifHMb6iFf1A.png)

#### input sequence
- 위 그림에서의 ABCD에 해당
    * tokenize -> index -> embed 순에서 index에 의해서 object -> 해당 object에 대한 번호로 숫자 전환
~~~
[“<pad>”, “<pad>”, “<pad>”, “Hello”, “, “, “how”, “are”, “you”, “?”] →
[5, 5, 5, 34, 90, 15, 684, 55, 193]
~~~

#### positional encoding
![positional encoding](https://miro.medium.com/max/700/1*V8ONEu6cph9Z8-QwaRHM-Q.png)
![sin cos](https://miro.medium.com/max/700/1*xCeAOFp17t-NcWWpF2k9Gw.png)

![positional encode](https://miro.medium.com/max/700/1*i4k32A-DJhdrtuB4Ty76Wg.png)

#### Scaled Dot-Product Attention
![attention 수식](https://image.slidesharecdn.com/attentionisallyouneed-190615112106/95/attention-is-all-you-need-13-638.jpg?cb=1560597686)

![qkv 관계](https://wikidocs.net/images/page/22893/%EC%BF%BC%EB%A6%AC.PNG)
- Attention 함수 뜯어보기

    * Key value 자료형의 구성은 Dictionary의 key value와 같음
    * Query Key와 곱한다는 말은 vector의 곱 (내적: dot product) -> Q와 K와의 유사도를 구하겠다
    * softmax X * V 라는 말은 X를 0~1로 확률값으로 scale해서 V에 반영하겠다.

 
- additive attention : 기본적인 linear
- scaled dot product : 내적
    * query key vlue 세 가지를 가지고 계산해 내는 것이 핵심

- self attention : Query와 key가 같은 경우
![self attention](https://heung-bae-lee.github.io/image/Scaled_Dot_Product_Attention_example.png)

- RNN은 sequence 길이가 길어질 수록 잊어버릴 가능성이 있다.
- Transformer의 self attention 기법은 query key가 같으므로써 모든 sequence간의 간계가 v에 항상 유지된다.

#### Multi-Head Attention
- multi head attention - scaled dot product attention 을 여러번 실행하는 것
![multihead attention heads](https://nlpinkorean.github.io/images/transformer/transformer_attention_heads_weight_matrix_o.png)
- self attention 계산과정을 여러번 거치게 되면 n개의 서로다른 z를 가지게 되고,
이후 각 head의 결과 행렬을 이어 붙여 긴 행렬을 만들어서 Wo 가중치를 내적해 계산하여 feed-forward 의 입력단으로 전달

![multi head attention](https://nlpinkorean.github.io/images/transformer/transformer_multi-headed_self-attention-recap.png)

#### Layer normalization

- input을 feature 단위로 normalize한다는 뜻. 
- batch normalization은 batch 단위로 normalize

![norm tech](https://miro.medium.com/max/512/1*F8KDxyfGG63QbJB2SB2aJw.png)

- residual을 사용. residual의 의미 = X + sub_layer(X) 
- 원 소스 내용의 감소를 방지하겠다.
- ![layer norm](https://miro.medium.com/max/290/1*HRX5QmV1viDj3DtjdbVlLQ.png)

#### Pointwise Feedforward Neetwork
- vector : (input sequence X hidden dimension)
- 2 linear layer + relu activation
FFNN(x)=MAX(0,xW1+b1)W2+b2
![pointwise](https://wikidocs.net/images/page/31379/positionwiseffnn.PNG)

#### Masked Self Attention
- Decoder는 Encoder와 동일하지만, shifted input으로 masked multi-head attention을 쓴다는 점이 다름
- Masked를 쓰는 이유는 self attention 시 자신의 timestep 이후 word는 가려 self attention이 되는 것을 막는 역할

"""
    This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
"""

- shifted input sequence : next token을 판단하기 위한 조치

#### Encoder Decoder Attention
decoder input과 encoder output간의 attention

![ed attention](https://miro.medium.com/max/623/1*y_oOzc5s7I6urwrXiIcQAA.png)
![query](https://miro.medium.com/max/500/1*RdiEz0jupMwiaoGMgHThEg.png)

- 디코딩 스텝은 decoder가 출력을 완료했다는 special 기호인 <end of sentence>를 출력할 때까지 반복

![decode](https://nlpinkorean.github.io/images/transformer/transformer_decoding_2.gif)
