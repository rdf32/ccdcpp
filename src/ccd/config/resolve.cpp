#include "ccd/config/resolve.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "ccd/input/procedure.hpp"

#include "convert.hpp"
#include "digest.hpp"

namespace ccd
{
namespace config
{

namespace
{

// The shared values resolved: as the run reads them, by name as the stages
// and steps read them, and the number of detection bands.
struct SharedValues
{
    Shared typed;
    Values named;
    index_t detection_bands = 0;
};

// The shared values of `given` against `input` (null: as written).
SharedValues resolve_shared(const Values& given, const Input* input)
{
    static const std::vector<ParameterInfo> info = info_of(shared_schema());
    SectionContext as_written;
    as_written.input = input;
    SharedValues out;
    apply(shared_schema(), section_values(info, given, "shared", as_written, "value"), out.typed,
          "shared");

    // A fit of min_observations observations (a start window, a whole-series
    // curve) has coefficient_min coefficients at least, and its RMSE needs a
    // degree of freedom.
    if (out.typed.min_observations <= out.typed.coefficient_min)
        throw std::invalid_argument(
            "shared values 'min_observations' and 'coefficient_min' are " +
            std::to_string(out.typed.min_observations) + " and " +
            std::to_string(out.typed.coefficient_min) +
            "; min_observations must exceed coefficient_min");

    const auto given_bands = given.find("detection_bands");
    const Value& bands = given_bands != given.end()
                             ? given_bands->second
                             : *std::find_if(info.begin(), info.end(), [](const ParameterInfo& p)
                                             { return p.name == "detection_bands"; })
                                    ->default_value;
    if (std::holds_alternative<std::string>(bands))
        throw std::invalid_argument("shared value 'detection_bands' is a list of roles or rows");
    if (const auto* roles = std::get_if<std::vector<std::string>>(&bands))
        out.detection_bands = roles->size();
    else if (const auto* rows = std::get_if<std::vector<std::int64_t>>(&bands))
        out.detection_bands = rows->size();
    if (out.detection_bands == 0)
        throw std::invalid_argument("shared value 'detection_bands' is empty");
    if (input)
        check_band_list(out.typed.detection_bands, input->bands(),
                        "shared value 'detection_bands'");
    out.named = to_values(shared_schema(), out.typed);
    return out;
}

// The stages of `stages` resolved `with`. A stage the algorithm does not have
// is refused as a strategy its stage does not have.
ConfigTree resolve_stages(const Registry& registry, const ConfigTree& stages,
                          const SectionContext& with)
{
    ConfigTree out;
    for (const auto& [stage, chosen] : stages)
    {
        const StrategyType& type = registry.strategy(stage, chosen.strategy);
        const std::string where = stage + " '" + chosen.strategy + "'";
        out[stage] = {chosen.strategy,
                      type.check(section_values(type.info, chosen.params, where, with),
                                 with.input, where)};
    }
    return out;
}

// The preprocessing of `preprocess` resolved `with`, every procedure
// required.
PreprocessConfig resolve_preprocess(const Registry& registry, const PreprocessConfig& preprocess,
                                    const SectionContext& with)
{
    for (const auto& [procedure, _] : preprocess)
        if (!procedure_named(procedure))
            throw std::invalid_argument("preprocess has no procedure '" + procedure +
                                        "'; the procedures are " + listed(procedures()));
    PreprocessConfig out;
    for (const std::string& procedure : procedures())
    {
        const auto it = preprocess.find(procedure);
        if (it == preprocess.end())
            throw std::invalid_argument("the configuration's preprocess has no '" + procedure +
                                        "' procedure");
        const ProcedureConfig& chain = it->second;
        if (chain.steps.size() > static_cast<std::size_t>(preprocess::CHAIN_CAPACITY))
            throw std::invalid_argument("preprocess " + procedure + " has " +
                                        std::to_string(chain.steps.size()) +
                                        " steps; a chain holds at most " +
                                        std::to_string(preprocess::CHAIN_CAPACITY));
        ProcedureConfig& resolved = out[procedure];
        resolved.thermal_transform = chain.thermal_transform;
        for (std::size_t i = 0; i < chain.steps.size(); ++i)
        {
            const StepConfig& step = chain.steps[i];
            const std::string at = StepContext{procedure, chain.thermal_transform, i}.where();
            const StepType& type = registry.step(step.step, at);
            const std::string where = at + " '" + step.step + "'";
            resolved.steps.push_back(
                {step.step, type.check(section_values(type.info, step.params, where, with),
                                       with.input, where)});
        }
    }
    return out;
}

const StageConfig& stage_of(const ConfigTree& config, const char* stage)
{
    const auto it = config.find(stage);
    if (it == config.end())
        throw std::invalid_argument(std::string("the configuration has no '") + stage +
                                    "' stage");
    return it->second;
}

// Every strategy's needs, from the stages before it.
void check_capabilities(const Strategies& s, const ConfigTree& config)
{
    const auto named = [&](const char* stage)
    { return std::string(stage) + " '" + config.at(stage).strategy + "'"; };
    const auto require = [](Capabilities needs, Capabilities provided, const std::string& who,
                            const std::string& providers)
    {
        const Capabilities missing = needs.missing_from(provided);
        if (!missing.empty())
            throw std::invalid_argument(who + " needs " + missing.names() + ", which " +
                                        providers + " does not provide");
    };
    require(s.stability_rule->needs(), s.stability_fit->provides(), named(StabilityRule::stage),
            named(StabilityFit::stage));
    require(s.lookback->needs(), s.stability_fit->provides(), named(Lookback::stage),
            named(StabilityFit::stage));
    const Capabilities model = s.model->provides();
    require(s.change_test->needs(), model, named(ChangeTest::stage), named(MonitoringModel::stage));
    require(s.outlier->needs(), model | s.change_test->provides(), named(OutlierRule::stage),
            named(MonitoringModel::stage) + " with " + named(ChangeTest::stage));
    const std::vector<std::string> provided = s.change_test->statistics();
    for (const std::string& statistic : s.outlier->statistics())
        if (std::find(provided.begin(), provided.end(), statistic) == provided.end())
            throw std::invalid_argument(named(OutlierRule::stage) + " reads the statistic '" +
                                        statistic + "', which " + named(ChangeTest::stage) +
                                        " does not provide" +
                                        (provided.empty() ? std::string()
                                                          : "; it provides " + listed(provided)));
}

// The plan of a resolved preprocessing on `input`, checked against its bands.
preprocess::PreprocessPlan build_plan(const Registry& registry, const PreprocessConfig& resolved,
                                      const Input& input)
{
    preprocess::PreprocessPlan plan;
    for (const auto& [procedure, chain] : resolved)
    {
        preprocess::ProcedurePlan& of = plan.of(*procedure_named(procedure));
        if (chain.thermal_transform)
            of.transform = preprocess::thermal_transform(input);
        for (std::size_t i = 0; i < chain.steps.size(); ++i)
        {
            const StepConfig& step = chain.steps[i];
            const StepContext context{procedure, chain.thermal_transform, i};
            const std::string at = context.where() + " '" + step.step + "'";
            if (auto built = registry.step(step.step, at).make(step.params, input, context, at))
                of.chain.push_back(*built);
        }
    }
    preprocess::validate(plan, input.bands());
    return plan;
}

// The Own values of `info`: given, or at their defaults, a whole number given
// for a real default as that real.
Values spelled_out(const std::vector<ParameterInfo>& info, const Values& given)
{
    Values out;
    for (const ParameterInfo& p : info)
    {
        if (p.source != Source::Own)
            continue;
        const auto it = given.find(p.name);
        Value value = it != given.end() ? it->second : *p.default_value;
        if (const auto* whole = std::get_if<std::int64_t>(&value))
            if (std::holds_alternative<double>(*p.default_value))
                value = real(static_cast<double>(*whole));
        out[p.name] = std::move(value);
    }
    return out;
}

} // namespace

Strategies build_strategies(const Registry& registry, const ConfigTree& config)
{
    Strategies s;
    s.mode = registry.make<ModeRule>(stage_of(config, ModeRule::stage));
    s.noise = registry.make<Noise>(stage_of(config, Noise::stage));
    s.start_search = registry.make<StartSearch>(stage_of(config, StartSearch::stage));
    s.window_screen = registry.make<WindowScreen>(stage_of(config, WindowScreen::stage));
    s.stability_fit = registry.make<StabilityFit>(stage_of(config, StabilityFit::stage));
    s.stability_rule = registry.make<StabilityRule>(stage_of(config, StabilityRule::stage));
    s.lookback = registry.make<Lookback>(stage_of(config, Lookback::stage));
    s.model = registry.make<MonitoringModel>(stage_of(config, MonitoringModel::stage));
    s.refit = registry.make<RefitSchedule>(stage_of(config, RefitSchedule::stage));
    s.change_test = registry.make<ChangeTest>(stage_of(config, ChangeTest::stage));
    s.outlier = registry.make<OutlierRule>(stage_of(config, OutlierRule::stage));
    s.segment_report = registry.make<SegmentReport>(stage_of(config, SegmentReport::stage));
    s.edge_segments = registry.make<EdgeSegments>(stage_of(config, EdgeSegments::stage));
    check_capabilities(s, config);
    return s;
}

void check(const Registry& registry, const Config& config)
{
    const SharedValues shared = resolve_shared(config.shared, nullptr);
    const SectionContext with{nullptr, shared.named, shared.detection_bands};
    build_strategies(registry, resolve_stages(registry, config.stages, with));
    resolve_preprocess(registry, config.preprocess, with);
}

Config canonical(const Registry& registry, const Config& config)
{
    check(registry, config);
    Config out = config;
    out.shared = spelled_out(info_of(shared_schema()), config.shared);
    for (auto& [stage, chosen] : out.stages)
        chosen.params = spelled_out(registry.strategy(stage, chosen.strategy).info, chosen.params);
    for (auto& [procedure, chain] : out.preprocess)
        for (std::size_t i = 0; i < chain.steps.size(); ++i)
        {
            StepConfig& step = chain.steps[i];
            const std::string at = StepContext{procedure, chain.thermal_transform, i}.where();
            step.params = spelled_out(registry.step(step.step, at).info, step.params);
        }
    return out;
}

Resolved resolve(const Registry& registry, const Config& config, const Input& input)
{
    Resolved out;
    const SharedValues shared = resolve_shared(config.shared, &input);
    const SectionContext with{&input, shared.named, shared.detection_bands};
    out.shared = shared.typed;
    out.stages = resolve_stages(registry, config.stages, with);
    out.strategies = build_strategies(registry, out.stages);
    out.preprocess = resolve_preprocess(registry, config.preprocess, with);
    out.plan = build_plan(registry, out.preprocess, input);
    out.digest = resolution_digest(input, shared.named, out.stages, out.preprocess);
    return out;
}

} // namespace config
} // namespace ccd
