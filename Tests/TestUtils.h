#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <assert.h>

template<class T, int size>
void load_vector(
	T arr[size],
	std::ifstream &file
)
{
	T element;
	for (int i = 0; i < size; i++)
	{
		std::string line;
		std::getline(file, line);
		std::stringstream stream;
		stream << line;
		stream >> element;
		arr[i] = element;
	}
}

template<class T, int depth, int size>
void load_stream(
	hls::stream<T> in_stream[size],
	std::ifstream &file
)
{
	T vec[size];
	for (int i = 0; i < depth; i++)
	{
		load_vector<T,size>(vec, file);
		for (int j = 0; j < size; j++)
		{
			in_stream[j].write(vec[j]);
		}
	}
}

template<class T, int length, int depth, int size>
void load_stream_array(
	hls::stream<T> in_stream[length][size],
	std::string &filename
)
{
	std::ifstream file(filename);
	for (int i = 0; i < length; i++)
	{
		load_stream<T, depth, size>(in_stream[i], file);
	}
}

template<class T, int depth, int size>
void compare_stream(
	hls::stream<T> in_stream[size],
	std::string* vec_filename,
	std::string* log_filename
)
{
	T item;
	std::ifstream file(*vec_filename);
	std::ofstream log(*log_filename);
	T num = 0;
	float error = 0;
	int mismatch_count = 0;
	float avg_error = 0;
	bool good_result = true;
	for (int j = 0; j < depth; j++)
	{

		for (int i = 0; i < size; i++)
		{
			in_stream[i].read(item);
			std::string line;
			std::stringstream stream;
			std::getline(file, line);
			stream << line;
			stream >> num;
			log << item << " " << num;

			if (item != num)
			{
				error = 0;

				if (num != 0)
				{
					error = ((float) (((num - item)) / num)) * 100;
				}
				else if (item != 0)
				{
					error = ((float) (((item- num)) / item)) * 100;
				}

				log << " -miss relative error: " << error << "%";
				avg_error +=  std::abs(error);
				mismatch_count++;
				good_result = false;
			}

			log << std::endl;
		}
	}

	avg_error = mismatch_count > 0 ? avg_error / mismatch_count : 0;

	if (good_result)
	{
		std::cout << "Test Passsed!" << std::endl;
			log << "Test Succeded with 0 mismatches!" << std::endl;
	}
	else
	{
		std::cout << "Number of mismatchs: " << mismatch_count
			<< " average relative error:  " << avg_error << std::endl;

		log << "Number of mismatchs: " << mismatch_count
			<< " average relative error:  " << avg_error << std::endl;
	}
}