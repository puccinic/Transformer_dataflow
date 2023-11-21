// Transformer_dataflow.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#include "TestConstants.h"
#include "Test.h"
const size_t token_length = 128;
const size_t sequence_length = 384;

enum Test_types
{
	Test_Activation,
	Test_AttHead,
	Test_Concat,
	Test_Encoder,
	Test_FeedForward,
	Test_LayerNorm,
	Test_Linear,
	Test_Mask,
	Test_MatAdd,
	Test_MatMul,
	Test_MultiHeadAtt,
	Test_Scale,
	Test_ScaleDotAtt,
	Test_SoftMax,
	Test_Transpose,
	Test_VecAdd
};

int main() {
	std::string input_filename[6] = {
		"input1.txt",
		"input2.txt",
		"input3.txt",
		"input4.txt",
		"input5.txt",
		"input6.txt"
	};
	
	std::string result_filename = "golden_result.txt";
	std::string log_filename = "log.txt";

	Test_types test = Test_SoftMax;

	switch (test) {
	case Test_Activation:
		test_activations<TYPE,ROWS,COLS>(&input_filename[0], &result_filename, &log_filename);
		break;
	case Test_AttHead:
		test_attHead<TYPE, ROWS, COLS, COLS / NUM_HEADS>(&input_filename[0],
			&input_filename[1], &input_filename[2], 
			&input_filename[3], &result_filename, &log_filename
		);
		break;
	case Test_Concat:
		test_concat<TYPE, ROWS, COLS>(&input_filename[0], &input_filename[1], &result_filename, &log_filename);
		break;
	case Test_Encoder:
		//TODO
		break;
	case Test_FeedForward:
		test_FF<TYPE, ROWS, HIDDEN, COLS>(&input_filename[0], &input_filename[1], &input_filename[2], 
			&input_filename[3], &input_filename[4], &result_filename, &log_filename
		);
		break;
	case Test_LayerNorm:
		test_layernorm<TYPE, ROWS, COLS>(EPSILON, &input_filename[0], &input_filename[1],
			&input_filename[2], &result_filename, &log_filename
		);
		break;
	case Test_Linear:
		test_linear<TYPE, ROWS, HIDDEN, COLS>(&input_filename[0], &input_filename[1], &input_filename[2],
		&result_filename, &log_filename
		);
		break;
	case Test_Mask:
		test_mask<TYPE, ROWS, COLS>(&input_filename[0], &input_filename[1], &result_filename, &log_filename);
		break;
	case Test_MatAdd:
		test_matadd<TYPE, ROWS, COLS>(&input_filename[0], &input_filename[1], &result_filename, &log_filename);
		break;
	case Test_MatMul:
		test_matmul<TYPE, ROWS, HIDDEN, COLS>(&input_filename[0], &input_filename[1], 
			&result_filename, &log_filename
		);
		break;
	case Test_MultiHeadAtt:
		test_multiheadatt<TYPE, NUM_HEADS, ROWS, COLS, COLS / NUM_HEADS>(&input_filename[0], 
			&input_filename[1], &input_filename[2], &input_filename[3], &input_filename[4], 
			&input_filename[5], &result_filename, &log_filename
		);
		break;
	case Test_Scale:
		test_scale<TYPE,ROWS,COLS>(&input_filename[0], SCALE_FACTOR, &result_filename, &log_filename);
		break;
	case Test_ScaleDotAtt:
		test_scaledotatt<TYPE, ROWS, COLS>(&input_filename[0], &input_filename[1], 
			&result_filename, &log_filename
		);
		break;
	case Test_SoftMax:
		test_softmax<TYPE, COLS>(&input_filename[0], &result_filename, &log_filename);
		break;
	case Test_Transpose:
		test_transpose<TYPE, ROWS, COLS>(&input_filename[0], &result_filename, &log_filename);
		break;
	case Test_VecAdd:
		test_vecadd<TYPE, COLS>(&input_filename[0], &input_filename[1], &result_filename, &log_filename);
		break;
	default:
		break;
	}
}

// Ejecutar programa: Ctrl + F5 o menú Depurar > Iniciar sin depurar
// Depurar programa: F5 o menú Depurar > Iniciar depuración

// Sugerencias para primeros pasos: 1. Use la ventana del Explorador de soluciones para agregar y administrar archivos
//   2. Use la ventana de Team Explorer para conectar con el control de código fuente
//   3. Use la ventana de salida para ver la salida de compilación y otros mensajes
//   4. Use la ventana Lista de errores para ver los errores
//   5. Vaya a Proyecto > Agregar nuevo elemento para crear nuevos archivos de código, o a Proyecto > Agregar elemento existente para agregar archivos de código existentes al proyecto
//   6. En el futuro, para volver a abrir este proyecto, vaya a Archivo > Abrir > Proyecto y seleccione el archivo .sln
