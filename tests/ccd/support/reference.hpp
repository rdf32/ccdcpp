#pragma once

//==============================================================================
//
// The reference CSVs of tests/ccd/data that other software wrote (pyccd,
// scikit-learn): a header line, then one row of numbers per line.
//
//==============================================================================

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace golden
{

// The rows of the reference CSV `filename` after its header, each as its
// values; empty lines are skipped.
inline std::vector<std::vector<double>> read_reference_rows(const std::string& filename)
{
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);
    if (!file)
        throw std::runtime_error("Unable to open " + filename);

    std::string line;
    std::getline(file, line);   // header

    std::vector<std::vector<double>> rows;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ','))
            row.push_back(std::stod(value));
        rows.push_back(std::move(row));
    }
    return rows;
}

} // namespace golden
