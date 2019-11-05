# BERT-CNN
Repositório para o TCC aplicando BERT a uma arquitetura CNN

### Setup

Para rodar o CNN usando Word2Vec, é necessário baixar o modelo pré-treinado do Word2Vec da Google (GoogleNews-vectors-negative300.bin) do site https://code.google.com/archive/p/word2vec/

Além disso, será necessário baixar os datasets:

MR: Baixar o arquivo indicado como "sentence polarity dataset v1.0" do site http://www.cs.cornell.edu/people/pabo/movie-review-data/ e descompactar a pasta "rt-polaritydata" na pasta do projeto.

TREC: Baixar os arquivos indicados como "training set 5" e "Test set" do site https://cogcomp.seas.upenn.edu/Data/QA/QC/ e colocar numa pasta "TREC-data" no mesmo diretório do projeto.

PortTwitter: Baixar o arquivo dos dados do site https://sites.google.com/site/miningbrgroup/home/resources, criar uma pasta "PortTwitter" no diretório do projeto, e descompactar o neste diretório.

Além disso, algumas bibliotecas deverão ser instaladas utilizando pip install, das quais: numpy, pandas, sklearn, gensim, tensorflow (ou tensorflow-gpu), tensorflow-hub, bert, bert-tensorflow

Não é necessário baixar o BERT diretamente, já que o script se encarrega de baixar e utilizar o módulo através das bibliotecas "bert" e "tensorflow-hub".

### Rodar o script

Com os dados e modelo Word2Vec baixados, podemos rodar os scripts. Para rodar o word2vec, é necessário rodar:

    python run_cnn_w2v.py nome_do_dataset
  
Onde o nome do dataset pode ser 'MR', 'TREC' ou 'PORT_TWITTER'. O script irá rodar todas as configurações (static, nonstatic, multichannel e rand) para o dataset escolhido. Os resultados serão salvos em formato .pickle na pasta output_w2v gerada pelo script.

Já para o BERT, o script é:

    python run_cnn_bert.py nome_do_dataset configuracao_utilizada
  
A configuração deverá ser 'STATIC', 'NONSTATIC' ou 'MULTICHANNEL'. Uma pasta específica será criada para cada dataset e configuração contendo os resultados.

### Avaliando os resultados

Ao terminar de rodar os scripts para os datasets desejados, é necessário rodar o script evaluate.py, que irá agregar os resultados em um arquivo evaluations.txt, que poderá ser lido por um humano.
