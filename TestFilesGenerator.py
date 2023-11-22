import sys
import numpy as np
import torch
from torch import nn
from typing import TypeAlias

ROWS =  10
COLS =  10
HIDDEN = 10
SCALE_FACTOR =  10
NUM_HEADS =  10
INT_HIGH = 128
INT_LOW = -128

FileName: TypeAlias = str

def printMatrix(mat: torch.Tensor, file: FileName) -> None:
  with open(file, "w") as f:
    for item in mat.flatten().tolist():
      print(item, file=f)


def activation(matIn: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = nn.functional.relu(input1)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)
  

def atthead():
  pass

def concat(matIn1: FileName, matIn2: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  input2 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = torch.cat((input1, input2), dim=1)
  printMatrix(input1, matIn1)
  printMatrix(input2, matIn2)
  printMatrix(output, matOut)

def encoder():
  pass

def feedForward():
  pass

def layerNorm(matIn: FileName, matWeight: FileName, matBias: FileName, matOut: FileName) -> None:
  input1 = torch.rand((ROWS, COLS))
  weight = torch.rand(ROWS)
  bias = torch.rand(ROWS)
  output = nn.functional.layer_norm(input1, (COLS,), weight, bias)
  printMatrix(input1, matIn)
  printMatrix(weight, matWeight)
  printMatrix(bias, matBias)
  printMatrix(output, matOut)

def linear(matIn: FileName, matWeights: FileName, matBias: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, HIDDEN))
  weight = torch.randint(INT_LOW, INT_HIGH, (COLS,HIDDEN))
  bias = torch.randint(INT_LOW, INT_HIGH, (COLS,))
  output = nn.functional.linear(input1,weight,bias)
  printMatrix(input1, matIn)
  printMatrix(torch.transpose(weight,0,1), matWeights)
  printMatrix(bias, matBias)
  printMatrix(output, matOut)

def mask(matIn: FileName, matMask: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  random_tensor = torch.randint(0, 2, size=(ROWS, COLS))
  output = input1 * random_tensor
  printMatrix(input1, matIn)
  printMatrix(random_tensor, matMask)
  printMatrix(output, matOut)

def matAdd(matA: FileName, matB: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  input2 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = input1 + input2
  printMatrix(input1, matA)
  printMatrix(input2, matB)
  printMatrix(output, matOut)

def matMul(matA: FileName, matB: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, HIDDEN))
  input2 = torch.randint(INT_LOW, INT_HIGH, (HIDDEN, COLS))
  output = torch.matmul(input1, input2)
  printMatrix(input1, matA)
  printMatrix(input2, matB)
  printMatrix(output, matOut)

def multiHeadAtt():
  pass

def scale(matIn: FileName, matOut: FileName) ->None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = torch.div(input1, SCALE_FACTOR, rounding_mode='trunc')
  printMatrix(input1, matIn)
  printMatrix(output, matOut)
  

def scaleDotAtt(matIn: FileName, matMask: FileName, matOut: FileName) -> None:
  input1 = torch.rand((ROWS, COLS))
  attn_mask = torch.randint(0, 2, size=(ROWS, ROWS))
  output = nn.functional.scaled_dot_product_attention(input1, input1, input1, attn_mask.bool())
  printMatrix(input1, matIn)
  printMatrix(output, matOut)
  printMatrix(attn_mask,matMask)

def softmax(matIn: FileName, matOut: FileName) -> None:
  input1 = torch.rand(COLS)
  output = nn.functional.softmax(input1, dim=-1)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)


def transpose(matIn: FileName, matOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = torch.transpose(input1,0,1)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)

def vecAdd(vecA:FileName, vecB: FileName, vecOut: FileName) -> None:
  input1 = torch.randint(INT_LOW, INT_HIGH, (COLS,))
  input2 = torch.randint(INT_LOW, INT_HIGH, (COLS,))
  output = input1 + input2
  printMatrix(input1, vecA)
  printMatrix(input2, vecB)
  printMatrix(output, vecOut)


#List of valid Arguments
#Ask why there is a difference regarding floating point operations in python and C
'''Test_Activation,
	Test_AttHead, #TODO
	Test_Concat, 
	Test_Encoder, #TODO 
	Test_FeedForward, #TODO
	Test_LayerNorm,
	Test_Linear,
	Test_Mask,
	Test_MatAdd,
	Test_MatMul,
	Test_MultiHeadAtt, #TODO
	Test_Scale,
	Test_ScaleDotAtt,
	Test_SoftMax,
	Test_Transpose,
	Test_VecAdd
'''

input_filename: list[FileName] = [
  "input1.txt",
  "input2.txt",
  "input3.txt",
  "input4.txt",
  "input5.txt",
  "input6.txt"
]

result_filename: FileName = "golden_result.txt"

test = sys.argv[1]
match test:
  case "Test_Activation":
    activation(input_filename[0], result_filename)
  case "Test_Concat":
    concat(input_filename[0], input_filename[1], result_filename)
  case "Test_LayerNorm":
    layerNorm(input_filename[0], input_filename[1],
              input_filename[2], result_filename)
  case "Test_Linear":
    linear(input_filename[0], input_filename[1], input_filename[2], result_filename)
  case "Test_Mask":
    mask(input_filename[0], input_filename[1], result_filename)
  case "Test_MatAdd":
    matAdd(input_filename[0], input_filename[1], result_filename)
  case "Test_MatMul":
    matMul(input_filename[0],input_filename[1],result_filename)
  case "Test_ScaleDotAtt":
    scaleDotAtt(input_filename[0], input_filename[1], result_filename)
  case "Test_SoftMax":
    softmax(input_filename[0],result_filename)
  case "Test_Transpose":
    transpose(input_filename[0],result_filename)
  case "Test_VecAdd":
    vecAdd(input_filename[0],input_filename[1],result_filename)
  case "Test_Scale":
    scale(input_filename[0], result_filename)