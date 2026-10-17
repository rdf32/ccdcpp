#include "resume.hpp"

#include <limits>
#include <stdexcept>
#include <string>

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

// record_columns()' status of a pixel that failed (CubeResult::failures).
constexpr std::uint8_t STATUS_FAILED = 2;

} // namespace

py::dict header_to_python(const ccd::ResumeHeader& header)
{
    py::dict d;
    d["format"] = header.format;
    d["results_version"] = header.results_version;
    d["options_digest"] = header.options_digest;
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
    header.options_digest = bits_of(d, "options_digest", what);
    header.series_end = field(d, "series_end", what).cast<std::int64_t>();
    header.statistics_end = field(d, "statistics_end", what).cast<std::int64_t>();
    return header;
}

py::dict record_to_python(const ccd::PixelRecord& r)
{
    py::dict d;
    d["input_fingerprint"] = r.input_fingerprint;
    d["procedure"] = static_cast<std::uint8_t>(r.procedure);
    d["exit"] = static_cast<std::uint8_t>(r.standard.exit);
    d["window_start"] = static_cast<std::int64_t>(r.standard.window.start);
    d["window_stop"] = static_cast<std::int64_t>(r.standard.window.stop);
    d["prev_break"] = static_cast<std::int64_t>(r.standard.prev_break);
    d["fit_stop"] = static_cast<std::int64_t>(r.standard.fit_stop);
    d["grow"] = r.standard.grow;
    d["num_coef"] = static_cast<std::uint8_t>(r.standard.num_coef);
    d["magnitude"] = py::array_t<ccd::scalar_t>(
        static_cast<py::ssize_t>(r.standard.magnitude.size()), r.standard.magnitude.data());
    d["final_segments"] = static_cast<std::int64_t>(r.standard.final_segments);
    d["excluded_days"] = py::array_t<std::int64_t>(
        static_cast<py::ssize_t>(r.excluded_days.size()), r.excluded_days.data());
    return d;
}

ccd::PixelRecord record_from_python(const py::dict& d)
{
    const char* what = "the resume record";
    ccd::PixelRecord r;
    r.input_fingerprint = bits_of(d, "input_fingerprint", what);
    r.procedure = static_cast<ccd::PixelProcedure>(count_of<std::uint8_t>(d, "procedure", what));
    r.standard.exit = static_cast<ccd::standard::Exit>(count_of<std::uint8_t>(d, "exit", what));
    r.standard.window = ccd::Window(
        count_of<std::uint32_t>(d, "window_start", what),
        count_of<std::uint32_t>(d, "window_stop", what));
    r.standard.prev_break = count_of<std::uint32_t>(d, "prev_break", what);
    r.standard.fit_stop = count_of<std::uint32_t>(d, "fit_stop", what);
    r.standard.grow = field(d, "grow", what).cast<bool>();
    r.standard.num_coef = count_of<std::uint8_t>(d, "num_coef", what);
    r.standard.magnitude = field(d, "magnitude", what).cast<std::vector<ccd::scalar_t>>();
    r.standard.final_segments = count_of<std::uint32_t>(d, "final_segments", what);
    r.excluded_days = field(d, "excluded_days", what).cast<std::vector<std::int64_t>>();
    return r;
}

std::vector<ccd::ChangeModel> segments_from_python(const py::list& models)
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

py::dict record_columns(
    const ccd::CubeResult& cube,
    index_t H,
    index_t W,
    index_t B,
    bool status
)
{
    const std::size_t N = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    std::vector<const ccd::PixelResult*> at(N, nullptr);
    std::size_t days = 0;
    for (const auto& px : cube.pixels)
    {
        at[px.row * W + px.col] = &px;
        days += px.record.excluded_days.size();
    }

    const auto n = static_cast<py::ssize_t>(N);
    py::array_t<std::int32_t> row(n), col(n);
    py::array_t<std::uint64_t> fingerprint(n);
    py::array_t<std::uint8_t> procedure(n), exit(n), num_coef(n), status_arr(n);
    py::array_t<std::int64_t> window_start(n), window_stop(n), prev_break(n), fit_stop(n),
        final_segments(n), offsets(n + 1), excluded(static_cast<py::ssize_t>(days));
    py::array_t<bool> grow(n);
    py::array_t<ccd::scalar_t> magnitude(std::vector<py::ssize_t>{n, static_cast<py::ssize_t>(B)});

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
    d["row"] = row; d["col"] = col;
    d["input_fingerprint"] = fingerprint;
    d["procedure"] = procedure; d["exit"] = exit;
    d["window_start"] = window_start; d["window_stop"] = window_stop;
    d["prev_break"] = prev_break; d["fit_stop"] = fit_stop;
    d["grow"] = grow; d["num_coef"] = num_coef; d["magnitude"] = magnitude;
    d["final_segments"] = final_segments;
    d["excluded_offsets"] = offsets; d["excluded_days"] = excluded;
    if (status)
        d["status"] = status_arr;
    return d;
}

ccd::CubeResumeFrom cube_from_python(
    const py::dict& header,
    const py::dict& records,
    const py::dict& segments,
    index_t H,
    index_t W,
    index_t B
)
{
    const std::size_t N = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    const char* what = "the resume records";

    ccd::CubeResumeFrom from;
    from.header = header_from_python(header);
    from.pixels.resize(N);
    from.segments.resize(N);

    const auto n = static_cast<py::ssize_t>(N);
    const auto b = static_cast<py::ssize_t>(B);
    const auto row = column<std::int64_t>(records, "row", what, n);
    const auto col = column<std::int64_t>(records, "col", what, n);
    const auto fingerprint = column<std::uint64_t>(records, "input_fingerprint", what, n);
    const auto procedure = column<std::uint8_t>(records, "procedure", what, n);
    const auto exit = column<std::uint8_t>(records, "exit", what, n);
    const auto window_start = column<std::int64_t>(records, "window_start", what, n);
    const auto window_stop = column<std::int64_t>(records, "window_stop", what, n);
    const auto prev_break = column<std::int64_t>(records, "prev_break", what, n);
    const auto fit_stop = column<std::int64_t>(records, "fit_stop", what, n);
    const auto grow = column<bool>(records, "grow", what, n);
    const auto num_coef = column<std::uint8_t>(records, "num_coef", what, n);
    const auto magnitude = column<ccd::scalar_t>(records, "magnitude", what, n, 2, b);
    const auto final_segments = column<std::int64_t>(records, "final_segments", what, n);
    const auto offsets = column<std::int64_t>(records, "excluded_offsets", what, n + 1);
    const auto excluded = column<std::int64_t>(
        records, "excluded_days", what, length_of(records, "excluded_days", what));

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
        ccd::PixelRecord& r = from.pixels[i];
        r.input_fingerprint = f(s);
        r.procedure = static_cast<ccd::PixelProcedure>(p(s));
        r.standard.exit = static_cast<ccd::standard::Exit>(e(s));
        if (ws(s) < 0 || we(s) < 0 || pb(s) < 0 || fs(s) < 0 || fin(s) < 0)
            throw std::invalid_argument(
                "the resume record of pixel " + std::to_string(i) +
                " has a negative position or count");
        r.standard.window = ccd::Window(static_cast<index_t>(ws(s)), static_cast<index_t>(we(s)));
        r.standard.prev_break = static_cast<index_t>(pb(s));
        r.standard.fit_stop = static_cast<index_t>(fs(s));
        r.standard.grow = g(s);
        r.standard.num_coef = nc(s);
        if (r.standard.exit == ccd::standard::Exit::SeriesEnd)
            for (index_t b = 0; b < B; ++b)
                r.standard.magnitude.push_back(m(s, static_cast<py::ssize_t>(b)));
        r.standard.final_segments = static_cast<index_t>(fin(s));
        for (std::int64_t k = o(s); k < o(s + 1); ++k)
            r.excluded_days.push_back(x(k));
    }

    // The stored segments, by pixel, in the order the columns hold them.
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
        from.segments[static_cast<std::size_t>(rv(k)) * W + static_cast<std::size_t>(cv(k))]
            .push_back(std::move(model));
    }
    return from;
}

py::dict resume_schema()
{
    py::dict d;
    for (const auto& [name, dtype] : std::initializer_list<std::pair<const char*, const char*>>{
             {"row", "int32"}, {"col", "int32"}, {"input_fingerprint", "uint64"},
             {"procedure", "uint8"}, {"exit", "uint8"}, {"window_start", "int64"},
             {"window_stop", "int64"}, {"prev_break", "int64"}, {"fit_stop", "int64"},
             {"grow", "bool"}, {"num_coef", "uint8"}, {"magnitude", "float64"},
             {"final_segments", "int64"}, {"excluded_offsets", "int64"},
             {"excluded_days", "int64"}, {"status", "uint8"}})
        d[name] = dtype;
    return d;
}
