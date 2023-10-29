// Transformer_dataflow.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#include "Test.h"

const size_t token_length = 128;
const size_t sequence_length = 384;

int main() {
    std::string matA = "matA.txt";
    std::string matB = "matB.txt";
    std::string matres = "matRes.txt";
    std::string log = "log_file.txt";
    test_matmul<int, sequence_length, token_length, token_length>(&matA, &matB, &matres, &log);
}

// Ejecutar programa: Ctrl + F5 o menú Depurar > Iniciar sin depurar
// Depurar programa: F5 o menú Depurar > Iniciar depuración

// Sugerencias para primeros pasos: 1. Use la ventana del Explorador de soluciones para agregar y administrar archivos
//   2. Use la ventana de Team Explorer para conectar con el control de código fuente
//   3. Use la ventana de salida para ver la salida de compilación y otros mensajes
//   4. Use la ventana Lista de errores para ver los errores
//   5. Vaya a Proyecto > Agregar nuevo elemento para crear nuevos archivos de código, o a Proyecto > Agregar elemento existente para agregar archivos de código existentes al proyecto
//   6. En el futuro, para volver a abrir este proyecto, vaya a Archivo > Abrir > Proyecto y seleccione el archivo .sln
