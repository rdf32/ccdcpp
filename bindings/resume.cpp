#include "resume.hpp"

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

template<typename T>
py::array_t<T> column(const py::dict& d, const char* name, const char* what)
{
    return py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
        field(d, name, what));
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
    header.format = field(d, "format", what).cast<std::uint16_t>();
    header.results_version = field(d, "results_version", what).cast<std::uint32_t>();
    header.options_digest = field(d, "options_digest", what).cast<std::uint64_t>();
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
    r.input_fingerprint = field(d, "input_fingerprint", what).cast<std::uint64_t>();
    r.procedure = static_cast<ccd::PixelProcedure>(field(d, "procedure", what).cast<int>());
    r.standard.exit = static_cast<ccd::standard::Exit>(field(d, "exit", what).cast<int>());
    r.standard.window = ccd::Window(
        field(d, "window_start", what).cast<index_t>(),
        field(d, "window_stop", what).cast<index_t>());
    r.standard.prev_break = field(d, "prev_break", what).cast<index_t>();
    r.standard.fit_stop = field(d, "fit_stop", what).cast<index_t>();
    r.standard.grow = field(d, "grow", what).cast<bool>();
    r.standard.num_coef = field(d, "num_coef", what).cast<index_t>();
    r.standard.magnitude = field(d, "magnitude", what).cast<std::vector<ccd::scalar_t>>();
    r.standard.final_segments = field(d, "final_segments", what).cast<index_t>();
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
        sv(s) = static_cast<std::uint8_t>(at[i] ? at[i]->status : ccd::ResumeStatus::Resumed);
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

    const auto fingerprint = column<std::uint64_t>(records, "input_fingerprint", what);
    const auto procedure = column<std::uint8_t>(records, "procedure", what);
    const auto exit = column<std::uint8_t>(records, "exit", what);
    const auto window_start = column<std::int64_t>(records, "window_start", what);
    const auto window_stop = column<std::int64_t>(records, "window_stop", what);
    const auto prev_break = column<std::int64_t>(records, "prev_break", what);
    const auto fit_stop = column<std::int64_t>(records, "fit_stop", what);
    const auto grow = column<bool>(records, "grow", what);
    const auto num_coef = column<std::uint8_t>(records, "num_coef", what);
    const auto magnitude = column<ccd::scalar_t>(records, "magnitude", what);
    const auto final_segments = column<std::int64_t>(records, "final_segments", what);
    const auto offsets = column<std::int64_t>(records, "excluded_offsets", what);
    const auto excluded = column<std::int64_t>(records, "excluded_days", what);

    if (static_cast<std::size_t>(fingerprint.size()) != N ||
        static_cast<std::size_t>(offsets.size()) != N + 1 ||
        magnitude.ndim() != 2 || static_cast<std::size_t>(magnitude.shape(0)) != N ||
        static_cast<index_t>(magnitude.shape(1)) != B)
        throw std::invalid_argument(
            "the resume records do not cover the cube's " + std::to_string(N) +
            " pixels and " + std::to_string(B) + " bands");

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
    const auto row = column<std::int32_t>(segments, "row", seg);
    const auto col = column<std::int32_t>(segments, "col", seg);
    const auto start_day = column<std::int64_t>(segments, "start_day", seg);
    const auto end_day = column<std::int64_t>(segments, "end_day", seg);
    const auto break_day = column<std::int64_t>(segments, "break_day", seg);
    const auto observation_count = column<std::int64_t>(segments, "observation_count", seg);
    const auto change_probability = column<ccd::scalar_t>(segments, "change_probability", seg);
    const auto curve_qa = column<std::int64_t>(segments, "curve_qa", seg);
    const auto intercept = column<ccd::scalar_t>(segments, "intercept", seg);
    const auto rmse = column<ccd::scalar_t>(segments, "rmse", seg);
    const auto mag = column<ccd::scalar_t>(segments, "magnitude", seg);
    const auto coefficients = column<ccd::scalar_t>(segments, "coefficients", seg);

    const auto rows = row.size();
    if (coefficients.ndim() != 3 || coefficients.shape(0) != rows ||
        static_cast<index_t>(coefficients.shape(1)) != B ||
        coefficients.shape(2) != static_cast<py::ssize_t>(ccd::CCD_MAX_COEFS) ||
        intercept.ndim() != 2 || intercept.shape(0) != rows ||
        static_cast<index_t>(intercept.shape(1)) != B)
        throw std::invalid_argument(
            "the stored segments' band arrays are not (segments, " + std::to_string(B) + ", ...)");

    const auto rv = row.unchecked<1>(); const auto cv = col.unchecked<1>();
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
