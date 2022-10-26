from torch import nn

class TextClassificationModel(nn.Module):

  def __init__(self, vocab_size, embed_dim, num_class):
    super(TextClassificationModel, self).__init__()
    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
    self.batchnorm1 = nn.BatchNorm1d(embed_dim)
    self.dropout = nn.Dropout(p=0.1)
    self.fc = nn.Linear(embed_dim, num_class)
    self.sigmoid=nn.Sigmoid() 
    self.init_weights()
    lr=0.001

  def init_weights(self):
    initrange = 0.5
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc.weight.data.uniform_(-initrange, initrange)
    self.fc.bias.data.zero_()

  def forward(self, text, offsets):
    embedded = self.embedding(text, offsets)
    #return self.sigmoid(self.fc(embedded))
    #x = self.batchnorm1(embedded)
    x = self.dropout(embedded)
    return self.sigmoid(self.fc(x).squeeze())