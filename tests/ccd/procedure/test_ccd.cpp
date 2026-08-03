#include "ccd/ccd.hpp"

#include <iostream>
#include <fstream>


struct TestData
{
    std::vector<std::int64_t> dates;
    std::vector<std::uint8_t> qas;
    std::vector<ccd::scalar_t> spectra_storage;
};

TestData read_data(
    const std::string& filename
)
{
    std::ifstream file(filename);

    if(!file)
    {
        throw std::runtime_error(
            "Unable to open file"
        );
    }


    std::vector<std::int64_t> dates;
    std::vector<std::uint8_t> qas;


    std::array<
        std::vector<ccd::scalar_t>,
        7
    > bands;


    std::string line;


    while(std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        // date
        std::getline(ss,value,',');
        dates.push_back(
            std::stoll(value)
        );

        // blue-green-red-nir-swir1-swir2-thermal
        for(ccd::index_t b=0;b<7;++b)
        {
            std::getline(ss,value,',');

            bands[b].push_back(
                std::stod(value)
            );
        }

        // qa
        std::getline(ss,value,',');

        qas.push_back(
            static_cast<std::uint8_t>(
                std::stoi(value)
            )
        );
    }


    TestData result;

    result.dates = dates;
    result.qas   = qas;

    const ccd::index_t T = dates.size();

    // same layout as numpy:
    //
    // spectra.shape = (7,T)
    //
    result.spectra_storage.resize(7*T);

    for(ccd::index_t b=0; b < 7; ++b)
    {
        for(ccd::index_t t = 0; t < T; ++t)
        {
            result.spectra_storage[
                b*T+t
            ] = bands[b][t];
        }
    }


    return result;
}


int main() {

    auto data = read_data("test_3657_3610_observations.csv");

    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        data.dates.data(),
        {data.dates.size()}
    );

    constexpr ccd::index_t num_bands = 7;
    auto spectral = ccd::ArrayView<ccd::scalar_t, 2>::contiguous(
        data.spectra_storage.data(),
        {num_bands, data.dates.size()}
    );

    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(
        data.qas.data(),
        {data.dates.size()}
    );

    auto t0 = ccd::Clock::now();
    ccd::HarmonicOptions hoptions;
    ccd::LassoOptions loptions;

    auto results = ccd::detect(
        dates,
        spectral,
        qas,
        hoptions,
        loptions
    );

    auto tf = ccd::Clock::now();

    print_change_models(
        results.models
    );

    std::cout << "Processing Mask: " << std::endl;
    for (std::size_t obs = 0; obs < results.mask.size(); ++obs)
    {
        std::cout << static_cast<int>(results.mask[obs]) << ", ";
    }
    std::cout << "\n";
    std::cout << "number of change models: " << results.models.size() << std::endl;
    std::cout << "cloud prob: " << results.cloud_prob << std::endl;
    std::cout << "snow prob: " << results.snow_prob << std::endl;
    std::cout << "water prob: " << results.water_prob << std::endl;
    std::cout << "clear prob: " << results.clear_prob << std::endl;

    std::cout << "Total CCD time: " 
        << ccd::elapsed_ms(t0, tf) 
        << " ms" << std::endl;
}