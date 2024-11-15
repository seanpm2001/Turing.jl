module DynamicPPLModelTests

using Test: @testset, @test
using Turing

# TODO(penelopeysm): Move this to DynamicPPL Test Utils module
# This function is defined inside DynamicPPL.jl/test/test_util.jl which
# means that it's not accessible from here
function test_setval!(model, chain; sample_idx=1, chain_idx=1)
    var_info = DynamicPPL.VarInfo(model)
    spl = DynamicPPL.SampleFromPrior()
    θ_old = var_info[spl]
    DynamicPPL.setval!(var_info, chain, sample_idx, chain_idx)
    θ_new = var_info[spl]
    @test θ_old != θ_new
    vals = DynamicPPL.values_as(var_info, OrderedDict)
    iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))
    for (n, v) in mapreduce(collect, vcat, iters)
        n = string(n)
        if Symbol(n) ∉ keys(chain)
            # Assume it's a group
            chain_val = vec(
                MCMCChains.group(chain, Symbol(n)).value[sample_idx, :, chain_idx]
            )
            v_true = vec(v)
        else
            chain_val = chain[sample_idx, n, chain_idx]
            v_true = v
        end

        @test v_true == chain_val
    end
end

@testset "model.jl" begin
    @testset "setval! & generated_quantities" begin
        @testset "$model" for model in DynamicPPL.TestUtils.DEMO_MODELS
            chain = sample(model, Prior(), 10)
            # A simple way of checking that the computation is determinstic: run twice and compare.
            res1 = generated_quantities(model, MCMCChains.get_sections(chain, :parameters))
            res2 = generated_quantities(model, MCMCChains.get_sections(chain, :parameters))
            @test all(res1 .== res2)
            test_setval!(model, MCMCChains.get_sections(chain, :parameters))
        end
    end

    @testset "value_iterator_from_chain" begin
        @testset "$model" for model in DynamicPPL.TestUtils.DEMO_MODELS
            chain = sample(model, Prior(), 10; progress=false)
            for (i, d) in enumerate(DynamicPPL.value_iterator_from_chain(model, chain))
                for vn in keys(d)
                    val = DynamicPPL.getvalue(d, vn)
                    for vn_leaf in DynamicPPL.varname_leaves(vn, val)
                        val_leaf = DynamicPPL.getvalue(d, vn_leaf)
                        @test val_leaf == chain[i, Symbol(vn_leaf), 1]
                    end
                end
            end
        end
    end
end

end
