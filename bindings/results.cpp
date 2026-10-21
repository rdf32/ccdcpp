#include "results.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace
{

using ccd::index_t;

// One field of a dict, refused with its name when missing.
py::object field(const py::dict& d, const char* name, const char* what)
{
    if (!d.contains(name))
        throw std::invalid_argument(std::string(what) + " has no \"" + name + "\"");
    return d[name];
}

// A column as a C-contiguous array of T, refused unless it is an array of
// `ndim` dimensions whose first is `length` (and second `width`, for 2-D and
// 3-D).
template<typename T>
py::array_t<T> column(
    const py::dict& d,
    const char* name,
    const char* what,
    py::ssize_t length,
    py::ssize_t ndim = 1,
    py::ssize_t width = -1
)
{
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
        field(d, name, what));
    if (!a)
        throw std::invalid_argument(
            std::string(what) + ": \"" + name + "\" is not an array of numbers");
    if (a.ndim() != ndim || a.shape(0) != length || (ndim > 1 && width >= 0 && a.shape(1) != width))
        throw std::invalid_argument(
            std::string(what) + ": \"" + name + "\" has the wrong shape (" +
            std::to_string(length) + " entries expected" +
            (ndim > 1 ? " of " + std::to_string(width) : std::string()) + ")");
    return a;
}

// The length of a column whose length is the data's.
py::ssize_t length_of(const py::dict& d, const char* name, const char* what)
{
    auto a = py::array::ensure(field(d, name, what));
    if (!a || a.ndim() < 1)
        throw std::invalid_argument(std::string(what) + ": \"" + name + "\" is not an array");
    return a.shape(0);
}

// A non-negative integer field that fits T.
template<typename T>
T count_of(const py::dict& d, const char* name, const char* what)
{
    const auto v = field(d, name, what).cast<long long>();
    if (v < 0 || static_cast<unsigned long long>(v) > std::numeric_limits<T>::max())
        throw std::invalid_argument(
            std::string(what) + ": \"" + name + "\" is " + std::to_string(v) + ", outside 0.." +
            std::to_string(static_cast<unsigned long long>(std::numeric_limits<T>::max())));
    return static_cast<T>(v);
}

// A uint64 field: Python's int, or a stored int64's bits (a uint64 written to
// a signed column comes back negative).
std::uint64_t bits_of(const py::dict& d, const char* name, const char* what)
{
    const py::int_ v = field(d, name, what).cast<py::int_>();
    if (v.cast<py::object>() < py::int_(0))
        return static_cast<std::uint64_t>(v.cast<long long>());
    return v.cast<std::uint64_t>();
}

ccd::BandFit band_from_python(const py::dict& b)
{
    ccd::BandFit fit;
    fit.intercept = field(b, "intercept", "a band").cast<ccd::scalar_t>();
    fit.rmse = field(b, "rmse", "a band").cast<ccd::scalar_t>();
    fit.magnitude = field(b, "magnitude", "a band").cast<ccd::scalar_t>();
    const auto coefficients = field(b, "coefficients", "a band").cast<std::vector<ccd::scalar_t>>();
    if (coefficients.size() != fit.coefficients.size())
        throw std::invalid_argument(
            "a band has " + std::to_string(coefficients.size()) + " coefficients; " +
            std::to_string(fit.coefficients.size()) + " expected");
    for (std::size_t c = 0; c < coefficients.size(); ++c)
        fit.coefficients[c] = coefficients[c];
    return fit;
}

// record_columns()' status of a pixel that failed (CubeOutput::failures).
constexpr std::uint8_t STATUS_FAILED = 2;

// A record column: its name, and T, the element type record_columns()
// writes.
template<typename T>
struct RecordColumn
{
    using type = T;
    const char* name;
};

// The record columns, in the order record_columns() writes them and
// resume_schema() lists them; records_from_columns() reads them by name.
// Each is reached by the index record_column() finds for its name.
constexpr std::tuple RECORD_COLUMNS{
    RecordColumn<std::int32_t>{"row"},
    RecordColumn<std::int32_t>{"col"},
    RecordColumn<std::uint64_t>{"input_fingerprint"},
    RecordColumn<std::uint8_t>{"procedure"},
    RecordColumn<std::uint8_t>{"exit"},
    RecordColumn<std::int64_t>{"window_start"},
    RecordColumn<std::int64_t>{"window_stop"},
    RecordColumn<std::int64_t>{"prev_break"},
    RecordColumn<std::int64_t>{"fit_stop"},
    RecordColumn<bool>{"grow"},
    RecordColumn<std::uint8_t>{"num_coef"},
    RecordColumn<ccd::scalar_t>{"magnitude"},
    RecordColumn<std::int64_t>{"final_segments"},
    RecordColumn<std::int64_t>{"excluded_offsets"},
    RecordColumn<std::int64_t>{"excluded_days"},
    RecordColumn<std::uint8_t>{"status"},
};

template<std::size_t... I>
constexpr std::size_t find_record_column(std::string_view name, std::index_sequence<I...>)
{
    std::size_t at = sizeof...(I);
    ((std::string_view(std::get<I>(RECORD_COLUMNS).name) == name ? (at = I, 0) : 0), ...);
    return at;
}

// The index of the record column `name`; a name the table does not hold
// does not compile.
constexpr std::size_t record_column(std::string_view name)
{
    constexpr std::size_t count = std::tuple_size_v<decltype(RECORD_COLUMNS)>;
    const std::size_t at = find_record_column(name, std::make_index_sequence<count>{});
    return at < count ? at : throw std::logic_error("no record column of that name");
}

constexpr std::size_t ROW = record_column("row");
constexpr std::size_t COL = record_column("col");
constexpr std::size_t INPUT_FINGERPRINT = record_column("input_fingerprint");
constexpr std::size_t PROCEDURE = record_column("procedure");
constexpr std::size_t EXIT = record_column("exit");
constexpr std::size_t WINDOW_START = record_column("window_start");
constexpr std::size_t WINDOW_STOP = record_column("window_stop");
constexpr std::size_t PREV_BREAK = record_column("prev_break");
constexpr std::size_t FIT_STOP = record_column("fit_stop");
constexpr std::size_t GROW = record_column("grow");
constexpr std::size_t NUM_COEF = record_column("num_coef");
constexpr std::size_t MAGNITUDE = record_column("magnitude");
constexpr std::size_t FINAL_SEGMENTS = record_column("final_segments");
constexpr std::size_t EXCLUDED_OFFSETS = record_column("excluded_offsets");
constexpr std::size_t EXCLUDED_DAYS = record_column("excluded_days");
constexpr std::size_t STATUS = record_column("status");

template<std::size_t I>
constexpr const char* name_of()
{
    return std::get<I>(RECORD_COLUMNS).name;
}

// Record column I, empty, in its element type.
template<std::size_t I>
auto new_column(std::vector<py::ssize_t> shape)
{
    using T = typename std::tuple_element_t<I, decltype(RECORD_COLUMNS)>::type;
    return py::array_t<T>(std::move(shape));
}

// The numpy dtype of an element type.
template<typename T>
constexpr const char* dtype_name()
{
    if constexpr (std::is_same_v<T, bool>)          return "bool";
    else if constexpr (std::is_same_v<T, std::uint8_t>)  return "uint8";
    else if constexpr (std::is_same_v<T, std::int32_t>)  return "int32";
    else if constexpr (std::is_same_v<T, std::int64_t>)  return "int64";
    else if constexpr (std::is_same_v<T, std::uint64_t>) return "uint64";
    else
    {
        static_assert(std::is_same_v<T, double>);
        return "float64";
    }
}

// A cube's (H, W, T) bytes as an array whose memory they become, without a
// copy.
py::array_t<std::uint8_t> cube_bytes(std::vector<std::uint8_t>&& bytes, index_t H, index_t W,
                                     index_t T)
{
    auto* held = new std::vector<std::uint8_t>(std::move(bytes));
    const py::capsule owner(
        held, [](void* p) { delete static_cast<std::vector<std::uint8_t>*>(p); });
    return py::array_t<std::uint8_t>(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(H), static_cast<py::ssize_t>(W),
                                 static_cast<py::ssize_t>(T)},
        held->data(), owner);
}

} // namespace

//------------------------------------------------------------------------------
// The resume header, as detect() and detect_cube() return it and resume=
// takes it back.
//------------------------------------------------------------------------------

py::dict header_to_python(const ccd::ResumeHeader& header)
{
    py::dict d;
    d["format"] = header.format;
    d["results_version"] = header.results_version;
    d["digest"] = header.digest;
    d["series_end"] = header.series_end;
    d["statistics_end"] = header.statistics_end;
    return d;
}

ccd::ResumeHeader header_from_python(const py::dict& d)
{
    const char* what = "the resume header";
    ccd::ResumeHeader header;
    header.format = count_of<std::uint16_t>(d, "format", what);
    header.results_version = count_of<std::uint32_t>(d, "results_version", what);
    ccd::check_version(header);   // before the fields another format may not hold
    header.digest = bits_of(d, "digest", what);
    header.series_end = field(d, "series_end", what).cast<std::int64_t>();
    header.statistics_end = field(d, "statistics_end", what).cast<std::int64_t>();
    return header;
}

//------------------------------------------------------------------------------
// One pixel's models, as detect() returns them: a list of dicts with a
// "bands" list each.
//------------------------------------------------------------------------------

py::list models_to_python(const std::vector<ccd::ChangeModel>& models)
{
    py::list out;
    for (const auto& model : models)
    {
        py::dict m;
        m["start_day"] = model.start_day;
        m["end_day"] = model.end_day;
        m["break_day"] = model.break_day;
        m["observation_count"] = model.observation_count;
        m["change_probability"] = model.change_probability;
        m["curve_qa"] = static_cast<std::uint8_t>(model.curve_qa);

        py::list bands;
        for (const auto& band : model.bands)
        {
            py::dict b;
            b["rmse"] = band.rmse;
            b["magnitude"] = band.magnitude;
            const auto& c = band.coefficients;
            py::tuple coef(c.size());
            for (std::size_t i = 0; i < c.size(); ++i)
                coef[i] = c[i];
            b["coefficients"] = coef;
            b["intercept"] = band.intercept;
            bands.append(b);
        }
        m["bands"] = bands;
        out.append(m);
    }
    return out;
}

std::vector<ccd::ChangeModel> models_from_python(const py::list& models)
{
    std::vector<ccd::ChangeModel> out;
    for (const py::handle h : models)
    {
        const py::dict m = h.cast<py::dict>();
        const char* what = "a stored segment";
        ccd::ChangeModel model;
        model.start_day = field(m, "start_day", what).cast<std::int64_t>();
        model.end_day = field(m, "end_day", what).cast<std::int64_t>();
        model.break_day = field(m, "break_day", what).cast<std::int64_t>();
        model.observation_count = field(m, "observation_count", what).cast<index_t>();
        model.change_probability = field(m, "change_probability", what).cast<ccd::scalar_t>();
        model.curve_qa = static_cast<ccd::CurveQA>(field(m, "curve_qa", what).cast<int>());
        for (const py::handle b : field(m, "bands", what).cast<py::list>())
            model.bands.push_back(band_from_python(b.cast<py::dict>()));
        out.push_back(std::move(model));
    }
    return out;
}

//------------------------------------------------------------------------------
// One pixel's resume record, as detect() returns it: a dict.
//------------------------------------------------------------------------------

py::dict record_to_python(const ccd::PixelRecord& r)
{
    py::dict d;
    d[name_of<INPUT_FINGERPRINT>()] = r.input_fingerprint;
    d[name_of<PROCEDURE>()] = static_cast<std::uint8_t>(r.procedure);
    d[name_of<EXIT>()] = static_cast<std::uint8_t>(r.standard.exit);
    d[name_of<WINDOW_START>()] = static_cast<std::int64_t>(r.standard.window.start);
    d[name_of<WINDOW_STOP>()] = static_cast<std::int64_t>(r.standard.window.stop);
    d[name_of<PREV_BREAK>()] = static_cast<std::int64_t>(r.standard.prev_break);
    d[name_of<FIT_STOP>()] = static_cast<std::int64_t>(r.standard.fit_stop);
    d[name_of<GROW>()] = r.standard.grow;
    d[name_of<NUM_COEF>()] = static_cast<std::uint8_t>(r.standard.num_coef);
    d[name_of<MAGNITUDE>()] = py::array_t<ccd::scalar_t>(
        static_cast<py::ssize_t>(r.standard.magnitude.size()), r.standard.magnitude.data());
    d[name_of<FINAL_SEGMENTS>()] = static_cast<std::int64_t>(r.standard.final_segments);
    d[name_of<EXCLUDED_DAYS>()] = py::array_t<std::int64_t>(
        static_cast<py::ssize_t>(r.excluded_days.size()), r.excluded_days.data());
    return d;
}

ccd::PixelRecord record_from_python(const py::dict& d)
{
    const char* what = "the resume record";
    ccd::PixelRecord r;
    r.input_fingerprint = bits_of(d, name_of<INPUT_FINGERPRINT>(), what);
    r.procedure = static_cast<ccd::PixelProcedure>(
        count_of<std::uint8_t>(d, name_of<PROCEDURE>(), what));
    r.standard.exit = static_cast<ccd::machine::Exit>(
        count_of<std::uint8_t>(d, name_of<EXIT>(), what));
    r.standard.window = ccd::Window(
        count_of<std::uint32_t>(d, name_of<WINDOW_START>(), what),
        count_of<std::uint32_t>(d, name_of<WINDOW_STOP>(), what));
    r.standard.prev_break = count_of<std::uint32_t>(d, name_of<PREV_BREAK>(), what);
    r.standard.fit_stop = count_of<std::uint32_t>(d, name_of<FIT_STOP>(), what);
    r.standard.grow = field(d, name_of<GROW>(), what).cast<bool>();
    r.standard.num_coef = count_of<std::uint8_t>(d, name_of<NUM_COEF>(), what);
    r.standard.magnitude =
        field(d, name_of<MAGNITUDE>(), what).cast<std::vector<ccd::scalar_t>>();
    r.standard.final_segments = count_of<std::uint32_t>(d, name_of<FINAL_SEGMENTS>(), what);
    r.excluded_days =
        field(d, name_of<EXCLUDED_DAYS>(), what).cast<std::vector<std::int64_t>>();
    return r;
}

//------------------------------------------------------------------------------
// detect_cube's columns
//
// One numpy array per field, one row per (pixel, segment), written into
// buffers allocated once: segments are counted first, then filled.
// ccdcpp.to_pixels() turns them into per-pixel dicts.
//
// Returned dict:
//
//   row, col              int32 [n]        cube-relative, 0-based
//   start_day, end_day,
//   break_day             int64 [n]        proleptic-Gregorian ordinals
//   curve_qa              uint8 [n]
//   observation_count     int32 [n]
//   change_probability  float64 [n]
//   intercept           float64 [n, B]     band-major
//   coefficients        float64 [n, B, 7]  (band, coefficient)
//   rmse                float64 [n, B]
//   magnitude           float64 [n, B]
//   num_bands             int              B, the input's band count
//   failures              list of (row, col, message)
//   threads               int              what actually ran
//   pixels_total          int              H * W
//   pixels_with_segments  int              pixels that produced >= 1 row
//   processing_mask       uint8 [H, W, T]  with masks=True only: [r, c] is
//                                          pixel (r, c)'s mask, 1 where the
//                                          observation is kept; zeros for a
//                                          failed pixel. The run's
//                                          CubeOutput::masks, not copied.
//   mask_reasons          uint8 [H, W, T]  with reasons=True only: [r, c] is
//                                          pixel (r, c)'s reason codes
//                                          (REASONS); REASON_COUNT for a
//                                          failed pixel. CubeOutput::reasons,
//                                          not copied.
//
// A pixel can produce zero segments, which is neither an error nor a
// failure: one whose observations are all fill, NaN, zero or cloud is in no
// array, and `failures` stays empty. A caller that needs every pixel
// present compares pixels_total with pixels_with_segments.
//
// Band order is the input's row order, B = Input.bands() rows; under the
// default layout that is blue, green, red, nir, swir1, swir2, thermal.
// Coefficient order is HarmonicBasis's column order (slope, cos1, sin1,
// cos2, sin2, cos3, sin3); the intercept is not among them, hence its own
// array. Coefficients above the model's own count are zero, not absent;
// zero is also a fitted value under L1, so a caller that needs to know which
// coefficients were fitted derives the count from curve_qa rather than
// testing values against zero.
//------------------------------------------------------------------------------

py::dict segment_columns(
    ccd::CubeOutput& cube,
    index_t H,
    index_t W,
    index_t B,
    index_t T,
    bool masks,
    bool reasons
)
{
    std::size_t n = 0;
    std::size_t with_segments = 0;
    for (const auto& pixel : cube.pixels)
    {
        n += pixel.result.models.size();
        if (!pixel.result.models.empty())
            ++with_segments;
    }

    const ccd::index_t NB = B;
    const auto nb = static_cast<py::ssize_t>(NB);
    constexpr ccd::index_t NC = ccd::CCD_MAX_COEFS;
    const auto rows = static_cast<py::ssize_t>(n);

    py::array_t<std::int32_t> row_arr(rows);
    py::array_t<std::int32_t> col_arr(rows);
    py::array_t<std::int64_t> start_arr(rows);
    py::array_t<std::int64_t> end_arr(rows);
    py::array_t<std::int64_t> break_arr(rows);
    py::array_t<std::uint8_t> qa_arr(rows);
    py::array_t<std::int32_t> obs_arr(rows);
    py::array_t<ccd::scalar_t> prob_arr(rows);
    py::array_t<ccd::scalar_t> int_arr(std::vector<py::ssize_t>{rows, nb});
    py::array_t<ccd::scalar_t> rmse_arr(std::vector<py::ssize_t>{rows, nb});
    py::array_t<ccd::scalar_t> magn_arr(std::vector<py::ssize_t>{rows, nb});
    py::array_t<ccd::scalar_t> coef_arr(std::vector<py::ssize_t>{rows, nb, NC});

    auto row_v = row_arr.mutable_unchecked<1>();
    auto col_v = col_arr.mutable_unchecked<1>();
    auto start_v = start_arr.mutable_unchecked<1>();
    auto end_v = end_arr.mutable_unchecked<1>();
    auto break_v = break_arr.mutable_unchecked<1>();
    auto qa_v = qa_arr.mutable_unchecked<1>();
    auto obs_v = obs_arr.mutable_unchecked<1>();
    auto prob_v = prob_arr.mutable_unchecked<1>();
    auto int_v = int_arr.mutable_unchecked<2>();
    auto rmse_v = rmse_arr.mutable_unchecked<2>();
    auto magn_v = magn_arr.mutable_unchecked<2>();
    auto coef_v = coef_arr.mutable_unchecked<3>();

    // Pixel-major in raster order (row-major), whatever the thread count, and
    // chronological within a pixel, the order the procedures produced them.
    py::ssize_t i = 0;
    for (const auto& pixel : cube.pixels)
    {
        for (const auto& model : pixel.result.models)
        {
            row_v(i) = static_cast<std::int32_t>(pixel.row);
            col_v(i) = static_cast<std::int32_t>(pixel.col);
            start_v(i) = model.start_day;
            end_v(i) = model.end_day;
            break_v(i) = model.break_day;
            qa_v(i) = static_cast<std::uint8_t>(model.curve_qa);
            obs_v(i) = static_cast<std::int32_t>(model.observation_count);
            prob_v(i) = model.change_probability;
            for (ccd::index_t b = 0; b < NB; ++b)
            {
                const auto& band = model.bands[b];
                int_v(i, b) = band.intercept;
                rmse_v(i, b) = band.rmse;
                magn_v(i, b) = band.magnitude;
                for (ccd::index_t c = 0; c < NC; ++c)
                    coef_v(i, b, c) = band.coefficients[c];
            }
            ++i;
        }
    }

    py::dict output;
    output["row"] = row_arr;
    output["col"] = col_arr;
    output["start_day"] = start_arr;
    output["end_day"] = end_arr;
    output["break_day"] = break_arr;
    output["curve_qa"] = qa_arr;
    output["observation_count"] = obs_arr;
    output["change_probability"] = prob_arr;
    output["intercept"] = int_arr;
    output["coefficients"] = coef_arr;
    output["rmse"] = rmse_arr;
    output["magnitude"] = magn_arr;
    output["num_bands"] = NB;
    output["failures"] = failures_to_python(cube.failures);
    output["threads"] = cube.threads;
    output["pixels_total"] =
        static_cast<std::int64_t>(H) * static_cast<std::int64_t>(W);
    output["pixels_with_segments"] = static_cast<std::int64_t>(with_segments);

    if (masks)
        output["processing_mask"] = cube_bytes(std::move(cube.masks), H, W, T);
    if (reasons)
        output["mask_reasons"] = cube_bytes(std::move(cube.reasons), H, W, T);
    return output;
}

std::vector<std::vector<ccd::ChangeModel>> segments_from_columns(
    const py::dict& segments,
    index_t H,
    index_t W,
    index_t B
)
{
    // By pixel, in the order the columns hold them.
    std::vector<std::vector<ccd::ChangeModel>> out(
        static_cast<std::size_t>(H) * static_cast<std::size_t>(W));
    const auto b = static_cast<py::ssize_t>(B);
    const char* seg = "the stored segments";
    const py::ssize_t rows = length_of(segments, "row", seg);
    const auto srow = column<std::int64_t>(segments, "row", seg, rows);
    const auto scol = column<std::int64_t>(segments, "col", seg, rows);
    const auto start_day = column<std::int64_t>(segments, "start_day", seg, rows);
    const auto end_day = column<std::int64_t>(segments, "end_day", seg, rows);
    const auto break_day = column<std::int64_t>(segments, "break_day", seg, rows);
    const auto observation_count = column<std::int64_t>(segments, "observation_count", seg, rows);
    const auto change_probability = column<ccd::scalar_t>(segments, "change_probability", seg, rows);
    const auto curve_qa = column<std::int64_t>(segments, "curve_qa", seg, rows);
    const auto intercept = column<ccd::scalar_t>(segments, "intercept", seg, rows, 2, b);
    const auto rmse = column<ccd::scalar_t>(segments, "rmse", seg, rows, 2, b);
    const auto mag = column<ccd::scalar_t>(segments, "magnitude", seg, rows, 2, b);
    const auto coefficients = column<ccd::scalar_t>(segments, "coefficients", seg, rows, 3, b);
    if (coefficients.shape(2) != static_cast<py::ssize_t>(ccd::CCD_MAX_COEFS))
        throw std::invalid_argument(
            "the stored segments: \"coefficients\" are not (segments, bands, " +
            std::to_string(ccd::CCD_MAX_COEFS) + ")");

    const auto rv = srow.unchecked<1>(); const auto cv = scol.unchecked<1>();
    const auto sd = start_day.unchecked<1>(); const auto ed = end_day.unchecked<1>();
    const auto bd = break_day.unchecked<1>(); const auto oc = observation_count.unchecked<1>();
    const auto cp = change_probability.unchecked<1>(); const auto cq = curve_qa.unchecked<1>();
    const auto iv = intercept.unchecked<2>(); const auto rm = rmse.unchecked<2>();
    const auto mg = mag.unchecked<2>(); const auto cf = coefficients.unchecked<3>();

    for (py::ssize_t k = 0; k < rows; ++k)
    {
        if (rv(k) < 0 || cv(k) < 0 || static_cast<index_t>(rv(k)) >= H ||
            static_cast<index_t>(cv(k)) >= W)
            throw std::invalid_argument(
                "a stored segment's pixel (" + std::to_string(rv(k)) + ", " +
                std::to_string(cv(k)) + ") is outside the cube");
        ccd::ChangeModel model;
        model.start_day = sd(k);
        model.end_day = ed(k);
        model.break_day = bd(k);
        model.observation_count = static_cast<index_t>(oc(k));
        model.change_probability = cp(k);
        model.curve_qa = static_cast<ccd::CurveQA>(cq(k));
        for (index_t b = 0; b < B; ++b)
        {
            const auto bb = static_cast<py::ssize_t>(b);
            ccd::BandFit fit;
            fit.intercept = iv(k, bb);
            fit.rmse = rm(k, bb);
            fit.magnitude = mg(k, bb);
            for (index_t c = 0; c < ccd::CCD_MAX_COEFS; ++c)
                fit.coefficients[c] = cf(k, bb, static_cast<py::ssize_t>(c));
            model.bands.push_back(fit);
        }
        out[static_cast<std::size_t>(rv(k)) * W + static_cast<std::size_t>(cv(k))]
            .push_back(std::move(model));
    }
    return out;
}

//------------------------------------------------------------------------------
// A cube's resume records, as detect_cube() returns them: columns over its
// H x W pixels in raster order.
//------------------------------------------------------------------------------

py::dict record_columns(
    const ccd::CubeOutput& cube,
    index_t H,
    index_t W,
    index_t B,
    bool status
)
{
    const std::size_t N = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    std::vector<const ccd::CubePixel*> at(N, nullptr);
    std::size_t days = 0;
    for (const auto& px : cube.pixels)
    {
        at[px.row * W + px.col] = &px;
        days += px.record.excluded_days.size();
    }

    const auto n = static_cast<py::ssize_t>(N);
    auto row = new_column<ROW>({n});
    auto col = new_column<COL>({n});
    auto fingerprint = new_column<INPUT_FINGERPRINT>({n});
    auto procedure = new_column<PROCEDURE>({n});
    auto exit = new_column<EXIT>({n});
    auto window_start = new_column<WINDOW_START>({n});
    auto window_stop = new_column<WINDOW_STOP>({n});
    auto prev_break = new_column<PREV_BREAK>({n});
    auto fit_stop = new_column<FIT_STOP>({n});
    auto grow = new_column<GROW>({n});
    auto num_coef = new_column<NUM_COEF>({n});
    auto magnitude = new_column<MAGNITUDE>({n, static_cast<py::ssize_t>(B)});
    auto final_segments = new_column<FINAL_SEGMENTS>({n});
    auto offsets = new_column<EXCLUDED_OFFSETS>({n + 1});
    auto excluded = new_column<EXCLUDED_DAYS>({static_cast<py::ssize_t>(days)});
    auto status_arr = new_column<STATUS>({n});

    auto rv = row.mutable_unchecked<1>(); auto cv = col.mutable_unchecked<1>();
    auto fv = fingerprint.mutable_unchecked<1>();
    auto pv = procedure.mutable_unchecked<1>(); auto ev = exit.mutable_unchecked<1>();
    auto nv = num_coef.mutable_unchecked<1>(); auto sv = status_arr.mutable_unchecked<1>();
    auto wsv = window_start.mutable_unchecked<1>(); auto wev = window_stop.mutable_unchecked<1>();
    auto bv = prev_break.mutable_unchecked<1>(); auto fsv = fit_stop.mutable_unchecked<1>();
    auto gv = grow.mutable_unchecked<1>(); auto mv = magnitude.mutable_unchecked<2>();
    auto finv = final_segments.mutable_unchecked<1>();
    auto ov = offsets.mutable_unchecked<1>(); auto xv = excluded.mutable_unchecked<1>();

    const ccd::PixelRecord none;
    std::int64_t next = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        const ccd::PixelRecord& r = at[i] ? at[i]->record : none;
        const auto s = static_cast<py::ssize_t>(i);
        rv(s) = static_cast<std::int32_t>(i / W);
        cv(s) = static_cast<std::int32_t>(i % W);
        fv(s) = r.input_fingerprint;
        pv(s) = static_cast<std::uint8_t>(r.procedure);
        ev(s) = static_cast<std::uint8_t>(r.standard.exit);
        nv(s) = static_cast<std::uint8_t>(r.standard.num_coef);
        sv(s) = at[i] ? static_cast<std::uint8_t>(at[i]->status) : STATUS_FAILED;
        wsv(s) = static_cast<std::int64_t>(r.standard.window.start);
        wev(s) = static_cast<std::int64_t>(r.standard.window.stop);
        bv(s) = static_cast<std::int64_t>(r.standard.prev_break);
        fsv(s) = static_cast<std::int64_t>(r.standard.fit_stop);
        gv(s) = r.standard.grow;
        finv(s) = static_cast<std::int64_t>(r.standard.final_segments);
        for (index_t b = 0; b < B; ++b)
            mv(s, static_cast<py::ssize_t>(b)) =
                b < r.standard.magnitude.size() ? r.standard.magnitude[b] : 0.0;
        ov(s) = next;
        for (const std::int64_t day : r.excluded_days)
            xv(next++) = day;
    }
    ov(n) = next;

    py::dict d;
    d[name_of<ROW>()] = row;
    d[name_of<COL>()] = col;
    d[name_of<INPUT_FINGERPRINT>()] = fingerprint;
    d[name_of<PROCEDURE>()] = procedure;
    d[name_of<EXIT>()] = exit;
    d[name_of<WINDOW_START>()] = window_start;
    d[name_of<WINDOW_STOP>()] = window_stop;
    d[name_of<PREV_BREAK>()] = prev_break;
    d[name_of<FIT_STOP>()] = fit_stop;
    d[name_of<GROW>()] = grow;
    d[name_of<NUM_COEF>()] = num_coef;
    d[name_of<MAGNITUDE>()] = magnitude;
    d[name_of<FINAL_SEGMENTS>()] = final_segments;
    d[name_of<EXCLUDED_OFFSETS>()] = offsets;
    d[name_of<EXCLUDED_DAYS>()] = excluded;
    if (status)
        d[name_of<STATUS>()] = status_arr;
    return d;
}

std::vector<ccd::PixelRecord> records_from_columns(
    const py::dict& records,
    index_t H,
    index_t W,
    index_t B
)
{
    const std::size_t N = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    const char* what = "the resume records";
    std::vector<ccd::PixelRecord> out(N);

    const auto n = static_cast<py::ssize_t>(N);
    const auto b = static_cast<py::ssize_t>(B);
    const auto row = column<std::int64_t>(records, name_of<ROW>(), what, n);
    const auto col = column<std::int64_t>(records, name_of<COL>(), what, n);
    const auto fingerprint = column<std::uint64_t>(records, name_of<INPUT_FINGERPRINT>(), what, n);
    const auto procedure = column<std::uint8_t>(records, name_of<PROCEDURE>(), what, n);
    const auto exit = column<std::uint8_t>(records, name_of<EXIT>(), what, n);
    const auto window_start = column<std::int64_t>(records, name_of<WINDOW_START>(), what, n);
    const auto window_stop = column<std::int64_t>(records, name_of<WINDOW_STOP>(), what, n);
    const auto prev_break = column<std::int64_t>(records, name_of<PREV_BREAK>(), what, n);
    const auto fit_stop = column<std::int64_t>(records, name_of<FIT_STOP>(), what, n);
    const auto grow = column<bool>(records, name_of<GROW>(), what, n);
    const auto num_coef = column<std::uint8_t>(records, name_of<NUM_COEF>(), what, n);
    const auto magnitude = column<ccd::scalar_t>(records, name_of<MAGNITUDE>(), what, n, 2, b);
    const auto final_segments = column<std::int64_t>(records, name_of<FINAL_SEGMENTS>(), what, n);
    const auto offsets = column<std::int64_t>(records, name_of<EXCLUDED_OFFSETS>(), what, n + 1);
    const auto excluded = column<std::int64_t>(
        records, name_of<EXCLUDED_DAYS>(), what, length_of(records, name_of<EXCLUDED_DAYS>(), what));

    {
        // Raster order, and offsets that slice excluded_days whole.
        const auto r = row.unchecked<1>(); const auto c = col.unchecked<1>();
        const auto o = offsets.unchecked<1>();
        const auto w = static_cast<py::ssize_t>(W);
        for (py::ssize_t k = 0; k < n; ++k)
            if (r(k) != k / w || c(k) != k % w)
                throw std::invalid_argument(
                    "the resume records are not in raster order: entry " + std::to_string(k) +
                    " is (" + std::to_string(r(k)) + ", " + std::to_string(c(k)) + ")");
        if (o(0) != 0 || o(n) != excluded.size())
            throw std::invalid_argument(
                "the resume records' excluded_offsets do not span excluded_days");
        for (py::ssize_t k = 0; k < n; ++k)
            if (o(k + 1) < o(k))
                throw std::invalid_argument(
                    "the resume records' excluded_offsets decrease at entry " + std::to_string(k));
    }

    const auto f = fingerprint.unchecked<1>(); const auto p = procedure.unchecked<1>();
    const auto e = exit.unchecked<1>(); const auto ws = window_start.unchecked<1>();
    const auto we = window_stop.unchecked<1>(); const auto pb = prev_break.unchecked<1>();
    const auto fs = fit_stop.unchecked<1>(); const auto g = grow.unchecked<1>();
    const auto nc = num_coef.unchecked<1>(); const auto m = magnitude.unchecked<2>();
    const auto fin = final_segments.unchecked<1>(); const auto o = offsets.unchecked<1>();
    const auto x = excluded.unchecked<1>();

    for (std::size_t i = 0; i < N; ++i)
    {
        const auto s = static_cast<py::ssize_t>(i);
        ccd::PixelRecord& r = out[i];
        r.input_fingerprint = f(s);
        r.procedure = static_cast<ccd::PixelProcedure>(p(s));
        r.standard.exit = static_cast<ccd::machine::Exit>(e(s));
        if (ws(s) < 0 || we(s) < 0 || pb(s) < 0 || fs(s) < 0 || fin(s) < 0)
            throw std::invalid_argument(
                "the resume record of pixel " + std::to_string(i) +
                " has a negative position or count");
        r.standard.window = ccd::Window(static_cast<index_t>(ws(s)), static_cast<index_t>(we(s)));
        r.standard.prev_break = static_cast<index_t>(pb(s));
        r.standard.fit_stop = static_cast<index_t>(fs(s));
        r.standard.grow = g(s);
        r.standard.num_coef = nc(s);
        if (r.standard.exit == ccd::machine::Exit::SeriesEnd)
            for (index_t b = 0; b < B; ++b)
                r.standard.magnitude.push_back(m(s, static_cast<py::ssize_t>(b)));
        r.standard.final_segments = static_cast<index_t>(fin(s));
        for (std::int64_t k = o(s); k < o(s + 1); ++k)
            r.excluded_days.push_back(x(k));
    }
    return out;
}

//------------------------------------------------------------------------------
// Failures, as Python data: a list of (row, col, message) tuples, empty in
// the normal case.
//------------------------------------------------------------------------------

py::list failures_to_python(const std::vector<ccd::CubeFailure>& failures)
{
    py::list out;
    for (const auto& f : failures)
        out.append(py::make_tuple(f.row, f.col, f.message));
    return out;
}

//------------------------------------------------------------------------------
// resume=: the earlier call's output.
//------------------------------------------------------------------------------

namespace
{

// `resume` as the earlier `entry` call's output, which has `has`.
py::dict previous_output(const py::object& resume, const char* entry, const char* has)
{
    if (!py::isinstance<py::dict>(resume))
        throw py::type_error(std::string(entry) + "(): resume= is the output of an earlier " +
                             entry + "() call, a dict");
    const py::dict out = resume.cast<py::dict>();
    if (!out.contains("resume") || !out.contains("resume_header"))
        throw py::value_error(std::string(entry) + "(): resume= holds no resume record; make "
                              "the earlier call with record=True");
    if (!out.contains(has))
        throw py::value_error(std::string(entry) + "(): resume= has no \"" + has +
                              "\"; it is the output of an earlier " + entry + "() call");
    return out;
}

} // namespace

std::optional<ccd::Resume> pixel_resume(const py::object& resume)
{
    if (resume.is_none())
        return std::nullopt;
    const py::dict previous = previous_output(resume, "detect", "models");
    return ccd::Resume{
        header_from_python(previous["resume_header"].cast<py::dict>()),
        {record_from_python(previous["resume"].cast<py::dict>())},
        {models_from_python(previous["models"].cast<py::list>())}};
}

std::optional<ccd::Resume> cube_resume(const py::object& resume, index_t H, index_t W, index_t B)
{
    if (resume.is_none())
        return std::nullopt;
    const py::dict previous = previous_output(resume, "detect_cube", "row");
    return ccd::Resume{
        header_from_python(previous["resume_header"].cast<py::dict>()),
        records_from_columns(previous["resume"].cast<py::dict>(), H, W, B),
        segments_from_columns(previous, H, W, B)};
}

//------------------------------------------------------------------------------
// Traces
//------------------------------------------------------------------------------

py::list trace_rows(const std::vector<ccd::ObserverEvent>& events)
{
    py::list out;
    for (const ccd::ObserverEvent& e : events)
    {
        py::dict d;
        d["kind"] = ccd::to_string(e.kind);
        d["machine"] = e.machine;
        d["phase"] = e.phase;
        d["next"] = e.next;
        d["window_start"] = e.window_start;
        d["window_stop"] = e.window_stop;
        d["prev_break"] = e.prev_break;
        d["observations"] = e.observations;
        d["day"] = e.day;
        d["day_to"] = e.day_to;
        d["outcome"] = e.outcome;
        py::list values;
        for (const ccd::scalar_t v : e.values)
            values.append(v);
        d["values"] = values;
        out.append(d);
    }
    return out;
}

py::dict resume_schema()
{
    py::dict d;
    std::apply(
        [&d](const auto&... column)
        { ((d[column.name] = dtype_name<typename std::decay_t<decltype(column)>::type>()), ...); },
        RECORD_COLUMNS);
    return d;
}
