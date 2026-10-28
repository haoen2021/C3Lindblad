#!/usr/bin/env julia

# Reuse only the quadrature, chain-map, serialization, and TDVP primitives from
# the accepted single-impurity driver. Its main routine is guarded by
# PROGRAM_FILE and is therefore not executed here.
include(joinpath(@__DIR__, "..", "single_impurity", "run.jl"))

using Random
using SHA

const DIMER_ROOT = pwd()

"""Geometry for two identical local baths and two additional impurity sites."""
function dimer_geometry(M::Integer)
    M > 0 || error("bath_spin_orbitals_total must be positive")
    M % 4 == 0 || error(
        "bath_spin_orbitals_total must be divisible by four: two spins and two baths"
    )
    m = M ÷ 4
    iseven(m) || error(
        "bath_spinful_sites_per_impurity must be even for the split Fermi-edge quadrature"
    )
    L = 2m + 2
    d1 = m + 1
    d2 = m + 2
    return (; M = Int(M), m, L, d1, d2)
end

spin_operators(spin) = spin == "up" ?
    ("Cdagup", "Cup", "Nup") : ("Cdagdn", "Cdn", "Ndn")

"""One-copy one-particle matrix in the MPS ordering documented in README.md."""
function dimer_one_body(M; td, W, Gamma, epsilon = 0.0)
    g = dimer_geometry(M)
    V, onsite, hopping = chain_parameters(g.m; W, Gamma)
    h = zeros(Float64, g.L, g.L)
    h[g.d1, g.d1] = epsilon
    h[g.d2, g.d2] = epsilon
    h[g.d1, g.d2] = h[g.d2, g.d1] = -td
    h[g.d1, g.d1 - 1] = h[g.d1 - 1, g.d1] = -V
    h[g.d2, g.d2 + 1] = h[g.d2 + 1, g.d2] = -V
    for k in 1:g.m
        h[g.d1 - k, g.d1 - k] = onsite[k]
        h[g.d2 + k, g.d2 + k] = onsite[k]
    end
    for k in 1:(g.m - 1)
        h[g.d1 - k, g.d1 - k - 1] = -hopping[k]
        h[g.d1 - k - 1, g.d1 - k] = -hopping[k]
        h[g.d2 + k, g.d2 + k + 1] = -hopping[k]
        h[g.d2 + k + 1, g.d2 + k] = -hopping[k]
    end
    return h
end

"""Spinful Anderson-dimer Hamiltonian with independent left and right baths."""
function build_dimer_hamiltonian(sites; M, U, td, W, Gamma)
    g = dimer_geometry(M)
    length(sites) == g.L || error("site count does not match dimer geometry")
    V, onsite, hopping = chain_parameters(g.m; W, Gamma)
    epsilon = -U / 2
    os = OpSum()
    for d in (g.d1, g.d2)
        os += epsilon, "Nup", d
        os += epsilon, "Ndn", d
        os += U, "Nupdn", d
    end
    for spin in ("up", "down")
        create, destroy, number = spin_operators(spin)
        os += -td, create, g.d1, destroy, g.d2
        os += -td, create, g.d2, destroy, g.d1
        os += -V, create, g.d1, destroy, g.d1 - 1
        os += -V, create, g.d1 - 1, destroy, g.d1
        os += -V, create, g.d2, destroy, g.d2 + 1
        os += -V, create, g.d2 + 1, destroy, g.d2
        for k in 1:g.m
            os += onsite[k], number, g.d1 - k
            os += onsite[k], number, g.d2 + k
        end
        for k in 1:(g.m - 1)
            os += -hopping[k], create, g.d1 - k, destroy, g.d1 - k - 1
            os += -hopping[k], create, g.d1 - k - 1, destroy, g.d1 - k
            os += -hopping[k], create, g.d2 + k, destroy, g.d2 + k + 1
            os += -hopping[k], create, g.d2 + k + 1, destroy, g.d2 + k
        end
    end
    return MPO(os, sites)
end

"""Half-filled product state with N_up=N_down=L/2."""
function dimer_initial_state(L)
    iseven(L) || error("the dimer chain must contain an even number of Electron sites")
    return [isodd(j) ? "Up" : "Dn" for j in 1:L]
end

"""
Apply a physical fermion at MPS site j.

In the ordered Electron tensor product, c_j is represented by
F_1 ... F_(j-1) c_j. This routine deliberately exposes that string rather
than relying on an MPO constructor to infer it.
"""
function fermion_apply(psi, sites, j, name)
    1 <= j <= length(sites) || error("fermion site lies outside the MPS")
    out = deepcopy(psi)
    for k in 1:(j - 1)
        out[k] = apply(op("F", sites[k]), out[k])
    end
    out[j] = apply(op(name, sites[j]), out[j])
    return out
end

function fermion_matrix_element(left, right, sites, j, name)
    return inner(left, fermion_apply(right, sites, j, name))
end

"""Exact noninteracting retarded impurity Green matrix for either spin."""
function exact_u0_green(t, M; td, W, Gamma)
    g = dimer_geometry(M)
    h = dimer_one_body(M; td, W, Gamma)
    impurity = [g.d1, g.d2]
    G = Array{ComplexF64}(undef, length(t), 2, 2)
    eig = eigen(Symmetric(h))
    Vd = eig.vectors[impurity, :]
    for (k, time) in enumerate(t)
        G[k, :, :] = -im .* (Vd * Diagonal(exp.(-im .* eig.values .* time)) * Vd')
    end
    return G
end

"""Impurity survival amplitude for one dimer parity channel in a star bath."""
function star_channel_amplitude(t; impurity_energy, m, W, Gamma)
    omega, weight = fermi_edge_quadrature(m; W, Gamma)
    h = zeros(Float64, m + 1, m + 1)
    h[1, 1] = impurity_energy
    h[1, 2:end] .= -sqrt.(weight)
    h[2:end, 1] .= -sqrt.(weight)
    h[2:end, 2:end] .= Diagonal(omega)
    eig = eigen(Symmetric(h))
    return [
        sum(abs2.(eig.vectors[1, :]) .* exp.(-im .* eig.values .* time))
        for time in t
    ]
end

"""
High-order evaluation of the analytic continuum U=0 dimer Green matrix.

The symmetric and antisymmetric impurity combinations have energies -td and
+td and each sees one copy of the analytic local bath. Increasing m changes
only this reference quadrature, not the finite benchmark bath.
"""
function continuum_u0_green(t; td, W, Gamma, m = 512)
    symmetric = star_channel_amplitude(
        t; impurity_energy = -td, m, W, Gamma
    )
    antisymmetric = star_channel_amplitude(
        t; impurity_energy = td, m, W, Gamma
    )
    G = zeros(ComplexF64, length(t), 2, 2)
    G[:, 1, 1] = G[:, 2, 2] = -im .* (symmetric + antisymmetric) ./ 2
    G[:, 1, 2] = G[:, 2, 1] = -im .* (symmetric - antisymmetric) ./ 2
    return G
end

function write_green_matrix(path, t, G)
    open(path, "w") do io
        fields = ["time"]
        for i in 1:2, j in 1:2
            push!(fields, "G$(i)$(j)_real", "G$(i)$(j)_imag")
        end
        println(io, join(fields, ','))
        for k in eachindex(t)
            row = Any[t[k]]
            for i in 1:2, j in 1:2
                push!(row, real(G[k, i, j]), imag(G[k, i, j]))
            end
            println(io, join(row, ','))
        end
    end
end

function window_error(observed, target, t, tmax)
    keep = findall(x -> x <= tmax + 100eps(Float64), t)
    tt = t[keep]
    rel, maximum_absolute = correlation_error(observed[keep], target[keep], tt)
    maximum_normalized = maximum_absolute / maximum(abs.(target[keep]))
    return Dict(
        "complex_relative_L2" => rel,
        "maximum_absolute" => maximum_absolute,
        "maximum_normalized" => maximum_normalized,
    )
end

function green_window_error(observed, target, t, tmax)
    keep = findall(x -> x <= tmax + 100eps(Float64), t)
    tt = t[keep]
    dt = diff(tt)
    squared_error = vec(sum(abs2.(observed[keep, :, :] - target[keep, :, :]); dims = (2, 3)))
    squared_target = vec(sum(abs2.(target[keep, :, :]); dims = (2, 3)))
    trapezoid(v) = sum(dt .* (v[1:(end - 1)] .+ v[2:end]) ./ 2)
    relative = sqrt(trapezoid(squared_error) / trapezoid(squared_target))
    maximum_absolute = maximum(abs.(observed[keep, :, :] - target[keep, :, :]))
    maximum_normalized = maximum_absolute / maximum(abs.(target[keep, :, :]))
    return Dict(
        "complex_relative_L2" => relative,
        "maximum_absolute" => maximum_absolute,
        "maximum_normalized" => maximum_normalized,
    )
end


function write_green_comparison(path, t, reference, finite)
    open(path, "w") do io
        fields = ["time"]
        for label in ("reference", "finite"), i in 1:2, j in 1:2
            push!(fields, "$(label)_G$(i)$(j)_real", "$(label)_G$(i)$(j)_imag")
        end
        println(io, join(fields, ','))
        for k in eachindex(t)
            row = Any[t[k]]
            for data in (reference, finite), i in 1:2, j in 1:2
                push!(row, real(data[k, i, j]), imag(data[k, i, j]))
            end
            println(io, join(row, ','))
        end
    end
end

function bath_preflight(config, output)
    model = config["model"]
    W = Float64(model["W"])
    Gamma = Float64(model["Gamma"])
    M_values = Int.(config["bath"]["bath_spin_orbitals_total"])
    td = Float64(model["td"])
    dt = Float64(config["bath"]["dt"])
    tmax = Float64(config["bath"]["tmax"])
    reference_m = Int(config["bath"]["reference_bath_spinful_sites_per_impurity"])
    t = collect(0.0:dt:tmax)
    target_l = target_correlation(t; W, Gamma, lesser = true)
    target_g = target_correlation(t; W, Gamma, lesser = false)
    continuum_green = continuum_u0_green(t; td, W, Gamma, m = reference_m)
    continuum_check = continuum_u0_green(t; td, W, Gamma, m = reference_m ÷ 2)
    finite_green = Dict{Int, Array{ComplexF64, 3}}()
    result = Dict{String, Any}(
        "temperature" => "beta=infinity",
        "chemical_potential" => 0.0,
        "W" => W,
        "Gamma" => Gamma,
        "bath_spin_orbitals_total_definition" =>
            "sum over both spins and both independent local baths",
    )
    for M in M_values
        g = dimer_geometry(M)
        finite_l = finite_bath_correlation(t; m = g.m, W, Gamma, lesser = true)
        finite_g = finite_bath_correlation(t; m = g.m, W, Gamma, lesser = false)
        finite_green[M] = exact_u0_green(t, M; td, W, Gamma)
        entry = Dict{String, Any}(
            "bath_spinful_sites_per_impurity" => g.m,
            "electron_mps_sites_including_dimer" => g.L,
        )
        for window in (0.5, 1.0)
            entry["lesser_tmax_$(window)"] = window_error(finite_l, target_l, t, window)
            entry["greater_tmax_$(window)"] = window_error(finite_g, target_g, t, window)
            entry["U0_green_tmax_$(window)"] = green_window_error(
                finite_green[M], continuum_green, t, window
            )
        end
        result["M$(M)"] = entry
        bath_output = joinpath(output, "M$(M)")
        mkpath(bath_output)
        write_bath_data(bath_output, t, target_l, target_g, finite_l, finite_g)
        write_green_comparison(
            joinpath(bath_output, "U0_green_matrix.csv"),
            t,
            continuum_green,
            finite_green[M],
        )
    end
    if length(M_values) >= 2
        first_M, second_M = M_values[1], M_values[2]
        pair = Dict{String, Any}()
        for window in (0.5, 1.0)
            pair["tmax_$(window)"] = green_window_error(
                finite_green[first_M], finite_green[second_M], t, window
            )
        end
        result["M$(first_M)_versus_M$(second_M)_U0_green"] = pair
    end
    convergence = Dict{String, Any}(
        "reference_bath_spinful_sites_per_impurity" => reference_m,
        "comparison_bath_spinful_sites_per_impurity" => reference_m ÷ 2,
    )
    for window in (0.5, 1.0)
        convergence["tmax_$(window)"] = green_window_error(
            continuum_check, continuum_green, t, window
        )
    end
    result["continuum_U0_quadrature_convergence"] = convergence
    write_toml(joinpath(output, "bath_preflight.toml"), result)
    return result
end

function u0_gate(config, output)
    model = config["model"]
    M = Int(model["bath_spin_orbitals_total"])
    td = Float64(model["td"])
    W = Float64(model["W"])
    Gamma = Float64(model["Gamma"])
    time = config["time"]
    t = collect(0.0:Float64(time["dt"]):Float64(time["tmax"]))
    G = exact_u0_green(t, M; td, W, Gamma)
    write_green_matrix(joinpath(output, "exact_U0_green_matrix.csv"), t, G)
    equal_time = G[1, :, :]
    result = Dict(
        "bath_spin_orbitals_total" => M,
        "bath_spinful_sites_per_impurity" => dimer_geometry(M).m,
        "electron_mps_sites_including_dimer" => dimer_geometry(M).L,
        "td" => td,
        "equal_time_maximum_error" => maximum(
            abs.(equal_time + im .* Matrix{ComplexF64}(I, 2, 2))
        ),
        "equal_time_offdiagonal_maximum" => maximum(abs.([equal_time[1, 2], equal_time[2, 1]])),
        "inversion_defect" => maximum(abs.(G[:, 1, 1] - G[:, 2, 2])),
        "reciprocity_defect" => maximum(abs.(G[:, 1, 2] - G[:, 2, 1])),
    )
    write_toml(joinpath(output, "exact_U0_gate.toml"), result)
    return result
end

function model_digest(; M, U, td, W, Gamma)
    g = dimer_geometry(M)
    V, onsite, hopping = chain_parameters(g.m; W, Gamma)
    text = join(
        (M, g.m, g.L, g.d1, g.d2, U, -U / 2, td, W, Gamma, V,
         join(onsite, ','), join(hopping, ','), "left-far-near-d1-d2-right-near-far"),
        '|',
    )
    return bytes2hex(sha256(text))
end

function resource_available(started, wall_cap_seconds, rss_cap_gib)
    wall_ok = time() - started <= wall_cap_seconds
    rss_ok = Sys.maxrss() <= rss_cap_gib * 2.0^30
    return wall_ok && rss_ok
end

function one_body_correlator(psi, sites, i, j, spin)
    create, destroy, _ = spin_operators(spin)
    acted = fermion_apply(psi, sites, j, destroy)
    acted = fermion_apply(acted, sites, i, create)
    return inner(psi, acted) / inner(psi, psi)
end

function two_site_expectation(psi, sites, d1, d2, name1, name2)
    os = OpSum()
    os += 1.0, name1, d1, name2, d2
    return inner(psi', MPO(os, sites), psi) / inner(psi, psi)
end

function ground_observables(psi, H, sites, g)
    norm2 = real(inner(psi, psi))
    energy = real(inner(psi', H, psi)) / norm2
    second_moment = real(inner(H, psi, H, psi)) / norm2
    variance = max(second_moment - energy^2, 0.0)
    impurity = Dict{String, Any}()
    for (label, d) in (("d1", g.d1), ("d2", g.d2))
        nup = real(expect(psi, "Nup"; sites = d))
        ndn = real(expect(psi, "Ndn"; sites = d))
        double = real(expect(psi, "Nupdn"; sites = d))
        impurity[label] = Dict(
            "n_up" => nup,
            "n_down" => ndn,
            "double_occupancy" => double,
            "local_moment" => nup + ndn - 2double,
        )
    end
    coherence = Dict(
        spin => complex_table(one_body_correlator(psi, sites, g.d1, g.d2, spin))
        for spin in ("up", "down")
    )
    return Dict(
        "state_norm_squared" => norm2,
        "energy" => energy,
        "energy_variance" => variance,
        "energy_standard_deviation" => sqrt(variance),
        "impurities" => impurity,
        "intersite_coherence_d1dag_d2" => coherence,
        "intersite_SzSz" => real(two_site_expectation(
            psi, sites, g.d1, g.d2, "Sz", "Sz"
        )),
    )
end

function compatible_ground_seed(saved; M, td, W, Gamma)
    return saved.M == M && saved.td == td && saved.W == W && saved.Gamma == Gamma &&
        saved.ordering == "left-far-near-d1-d2-right-near-far"
end

function dimer_ground(config, output)
    model = config["model"]
    dmrg_config = config["dmrg"]
    M = Int(model["bath_spin_orbitals_total"])
    U = Float64(model["U"])
    td = Float64(model["td"])
    W = Float64(model["W"])
    Gamma = Float64(model["Gamma"])
    digest = model_digest(; M, U, td, W, Gamma)
    g = dimer_geometry(M)
    sites = siteinds("Electron", g.L; conserve_qns = true)
    H = build_dimer_hamiltonian(sites; M, U, td, W, Gamma)
    Random.seed!(Int(dmrg_config["seed"]))
    initial_checkpoint = String(get(dmrg_config, "initial_checkpoint", ""))
    if isempty(initial_checkpoint)
        psi0 = random_mps(
            sites,
            dimer_initial_state(g.L);
            linkdims = Int(dmrg_config["initial_link_dimension"]),
        )
    else
        saved = open(deserialize, joinpath(DIMER_ROOT, initial_checkpoint))
        compatible_ground_seed(saved; M, td, W, Gamma) ||
            error("initial ground checkpoint has a different bath, geometry, or ordering")
        sites = saved.sites
        H = build_dimer_hamiltonian(sites; M, U, td, W, Gamma)
        psi0 = deepcopy(saved.psi)
    end
    nsweeps = Int(dmrg_config["sweeps"])
    maximum_dimension = Int(dmrg_config["maxdim"])
    schedule = min.(Int.(dmrg_config["maxdim_schedule"]), maximum_dimension)
    length(schedule) == nsweeps || error("maxdim_schedule length must equal sweeps")
    noise = Float64.(dmrg_config["noise_schedule"])
    length(noise) == nsweeps || error("noise_schedule length must equal sweeps")
    cutoff = Float64(dmrg_config["cutoff"])
    observer = DMRGObserver(
        ; energy_tol = Float64(dmrg_config["energy_tolerance"]), minsweeps = 4
    )
    started = time()
    energy, psi = dmrg(
        H,
        psi0;
        nsweeps,
        maxdim = schedule,
        cutoff,
        noise,
        observer,
        outputlevel = Int(dmrg_config["output_level"]),
    )
    wall = time() - started
    resource_cap_passed = wall <= Float64(dmrg_config["wall_cap_seconds"]) &&
        Sys.maxrss() <= Float64(dmrg_config["rss_cap_gib"]) * 2.0^30
    observables = ground_observables(psi, H, sites, g)
    checkpoint = joinpath(output, "ground_state.jls")
    package_versions = Dict(
        "julia" => string(VERSION),
        "ITensors" => string(Base.pkgversion(ITensors)),
        "ITensorMPS" => string(Base.pkgversion(ITensorMPS)),
    )
    open(checkpoint, "w") do io
        serialize(
            io,
            (; sites, psi, energy, model_digest = digest, M, U, epsilon = -U / 2,
             td, W, Gamma, ordering = "left-far-near-d1-d2-right-near-far",
             particle_sector = (N_up = g.L ÷ 2, N_down = g.L ÷ 2),
             package_versions),
        )
    end
    result = Dict(
        "model_digest" => digest,
        "bath_spin_orbitals_total" => M,
        "bath_spinful_sites_per_impurity" => g.m,
        "electron_mps_sites_including_dimer" => g.L,
        "U" => U,
        "epsilon" => -U / 2,
        "td" => td,
        "particle_sector" => Dict("N_up" => g.L ÷ 2, "N_down" => g.L ÷ 2),
        "requested_maximum_bond_dimension" => maximum_dimension,
        "initial_checkpoint" => initial_checkpoint,
        "actual_maximum_bond_dimension" => maxlinkdim(psi),
        "energy_by_sweep" => Float64.(real.(energies(observer))),
        "maximum_discarded_weight_by_sweep" => Float64.(truncerrors(observer)),
        "observables" => observables,
        "wall_seconds" => wall,
        "peak_rss_bytes" => Sys.maxrss(),
        "resource_cap_passed" => resource_cap_passed,
        "checkpoint" => relpath(checkpoint, DIMER_ROOT),
    )
    write_toml(joinpath(output, "ground_state.toml"), result)
    return result
end

function particle_hole_signs()
    eta = (1.0, -1.0)
    return [eta[i] * eta[j] for i in 1:2, j in 1:2]
end

function reconstruct_removal(addition)
    size(addition, 2) == 2 && size(addition, 3) == 2 ||
        error("the addition array must contain a 2x2 impurity block")
    signs = particle_hole_signs()
    removal = similar(addition)
    for sigma in axes(addition, 4), i in 1:2, j in 1:2
        removal[:, i, j, sigma] = -signs[i, j] .* conj.(addition[:, i, j, sigma])
    end
    return removal
end

function channel_defects(addition, removal)
    signs = particle_hole_signs()
    ph = 0.0
    for sigma in axes(addition, 4), i in 1:2, j in 1:2
        ph = max(ph, maximum(abs.(
            removal[:, i, j, sigma] + signs[i, j] .* conj.(addition[:, i, j, sigma])
        )))
    end
    spin = max(
        maximum(abs.(addition[:, :, :, 1] - addition[:, :, :, 2])),
        maximum(abs.(removal[:, :, :, 1] - removal[:, :, :, 2])),
    )
    total = dropdims(mean(addition + removal; dims = 4); dims = 4)
    equal_time = maximum(abs.(
        total[1, :, :] + im .* Matrix{ComplexF64}(I, 2, 2)
    ))
    inversion = max(
        maximum(abs.(total[:, 1, 1] - total[:, 2, 2])),
        maximum(abs.(total[:, 1, 2] - total[:, 2, 1])),
    )
    return (; particle_hole = ph, spin, equal_time, inversion, total)
end

function write_response_channels(path, t, addition, removal)
    open(path, "w") do io
        fields = ["time"]
        for branch in ("addition", "removal"), spin in ("up", "down"), i in 1:2, j in 1:2
            push!(fields, "$(branch)_$(spin)_$(i)$(j)_real", "$(branch)_$(spin)_$(i)$(j)_imag")
        end
        println(io, join(fields, ','))
        for k in eachindex(t)
            row = Any[t[k]]
            for data in (addition, removal), sigma in 1:2, i in 1:2, j in 1:2
                push!(row, real(data[k, i, j, sigma]), imag(data[k, i, j, sigma]))
            end
            println(io, join(row, ','))
        end
    end
end

function save_response_checkpoint(directory, completed_step, t, neutral, states, digest)
    mkpath(directory)
    open(joinpath(directory, "neutral.jls"), "w") do io
        serialize(io, (; completed_step, time = t[completed_step], state = neutral, model_digest = digest))
    end
    for (name, state) in states
        open(joinpath(directory, "source_$(name).jls"), "w") do io
            serialize(
                io,
                (; completed_step, time = t[completed_step], state,
                 model_digest = digest, source = name),
            )
        end
    end
    write_toml(
        joinpath(directory, "progress.toml"),
        Dict("completed_step" => completed_step, "time" => t[completed_step], "model_digest" => digest),
    )
end

function response_sources(psi, sites, g, mode)
    states = Dict{String, MPS}()
    spins = mode == "all_eight" ? ("up", "down") : ("up",)
    branches = mode == "all_eight" ? ("addition", "removal") : ("addition",)
    for branch in branches, spin in spins, j in 1:2
        create, destroy, _ = spin_operators(spin)
        name = branch == "addition" ? create : destroy
        states["$(branch)_$(spin)_$(j)"] = fermion_apply(
            psi, sites, j == 1 ? g.d1 : g.d2, name
        )
    end
    return states
end

function measure_response!(addition, removal, k, neutral, states, sites, g, mode)
    impurity = (g.d1, g.d2)
    spins = ("up", "down")
    for (sigma, spin) in enumerate(spins), source in 1:2, sink in 1:2
        _, destroy, _ = spin_operators(spin)
        add_key = "addition_$(spin)_$(source)"
        if haskey(states, add_key)
            addition[k, sink, source, sigma] = -im * fermion_matrix_element(
                neutral, states[add_key], sites, impurity[sink], destroy
            )
        end
        rem_key = "removal_$(spin)_$(source)"
        if haskey(states, rem_key)
            removal[k, sink, source, sigma] = -im * fermion_matrix_element(
                states[rem_key], neutral, sites, impurity[sink], destroy
            )
        end
    end
    if mode == "spin_up_addition_ph"
        addition[k, :, :, 2] = addition[k, :, :, 1]
    end
end

function validate_identity_gate(path, digest, tolerance)
    isempty(path) && error("reduced response mode requires an all-eight identity gate")
    gate = TOML.parsefile(joinpath(DIMER_ROOT, path))
    gate["model_digest"] == digest || error("identity gate has a different model digest")
    Float64(gate["identity_gate_tolerance"]) <= tolerance ||
        error("identity gate used a weaker tolerance")
    return Bool(gate["identity_gate_passed"])
end

function effective_response_mode(requested_mode, reduction_gate_validated)
    requested_mode in ("all_eight", "spin_up_addition_ph") ||
        error("branch_mode must be all_eight or spin_up_addition_ph")
    return requested_mode == "spin_up_addition_ph" && !reduction_gate_validated ?
        "all_eight" : requested_mode
end

function dimer_response(config, output)
    model = config["model"]
    tdvp_config = config["tdvp"]
    M = Int(model["bath_spin_orbitals_total"])
    U = Float64(model["U"])
    td = Float64(model["td"])
    W = Float64(model["W"])
    Gamma = Float64(model["Gamma"])
    digest = model_digest(; M, U, td, W, Gamma)
    g = dimer_geometry(M)
    ground_path = joinpath(DIMER_ROOT, String(tdvp_config["ground_checkpoint"]))
    saved = open(deserialize, ground_path)
    saved.model_digest == digest || error("ground checkpoint has a different model digest")
    sites = saved.sites
    psi = deepcopy(saved.psi)
    H = build_dimer_hamiltonian(sites; M, U, td, W, Gamma)
    maxdim = Int(tdvp_config["maxdim"])
    truncate!(psi; cutoff = Float64(tdvp_config["cutoff"]), maxdim)
    psi[1] *= inv(norm(psi))
    requested_mode = String(tdvp_config["branch_mode"])
    tolerance = Float64(tdvp_config["identity_tolerance"])
    reduction_gate_validated = requested_mode == "spin_up_addition_ph" &&
        validate_identity_gate(String(tdvp_config["identity_gate"]), digest, tolerance)
    mode = effective_response_mode(requested_mode, reduction_gate_validated)
    dt = Float64(tdvp_config["dt"])
    tmax = Float64(tdvp_config["tmax"])
    nstep = round(Int, tmax / dt)
    isapprox(nstep * dt, tmax; atol = 100eps(Float64), rtol = 0.0) ||
        error("tmax must be an integer multiple of dt")
    t = collect(0.0:dt:tmax)
    cutoff = Float64(tdvp_config["cutoff"])
    states = response_sources(psi, sites, g, mode)
    neutral = deepcopy(psi)
    addition = zeros(ComplexF64, length(t), 2, 2, 2)
    removal = zeros(ComplexF64, length(t), 2, 2, 2)
    maximum_discarded = Dict(name => 0.0 for name in keys(states))
    cumulative_discarded = Dict(name => 0.0 for name in keys(states))
    maximum_bond = Dict(name => maxlinkdim(state) for (name, state) in states)
    wall_by_state = Dict(name => 0.0 for name in keys(states))
    neutral_maximum_discarded = 0.0
    neutral_cumulative_discarded = 0.0
    neutral_maximum_bond = maxlinkdim(neutral)
    neutral_wall = 0.0
    neutral_norm = zeros(Float64, length(t))
    neutral_energy = zeros(Float64, length(t))
    source_norm = Dict(name => zeros(Float64, length(t)) for name in keys(states))
    checkpoint_directory = joinpath(output, "checkpoints")
    started = time()
    for k in eachindex(t)
        neutral_norm[k] = real(inner(neutral, neutral))
        neutral_energy[k] = real(inner(neutral', H, neutral)) / neutral_norm[k]
        for (name, state) in states
            source_norm[name][k] = real(inner(state, state))
        end
        measure_response!(addition, removal, k, neutral, states, sites, g, mode)
        k == lastindex(t) && break
        neutral, trunc, cumulative, _, bond, wall = tdvp_step(
            H,
            neutral;
            dt,
            cutoff,
            maxdim,
            nsite = 2,
            updater_krylovdim = Int(tdvp_config["updater_krylov_dimension"]),
            updater_tolerance = Float64(tdvp_config["updater_tolerance"]),
        )
        neutral_maximum_discarded = max(neutral_maximum_discarded, trunc)
        neutral_cumulative_discarded += cumulative
        neutral_maximum_bond = max(neutral_maximum_bond, bond)
        neutral_wall += wall
        for name in keys(states)
            states[name], trunc, cumulative, _, bond, wall = tdvp_step(
                H,
                states[name];
                dt,
                cutoff,
                maxdim,
                nsite = 2,
                updater_krylovdim = Int(tdvp_config["updater_krylov_dimension"]),
                updater_tolerance = Float64(tdvp_config["updater_tolerance"]),
            )
            maximum_discarded[name] = max(maximum_discarded[name], trunc)
            cumulative_discarded[name] += cumulative
            maximum_bond[name] = max(maximum_bond[name], bond)
            wall_by_state[name] += wall
        end
        save_response_checkpoint(checkpoint_directory, k + 1, t, neutral, states, digest)
        resource_available(
            started,
            Float64(tdvp_config["wall_cap_seconds"]),
            Float64(tdvp_config["rss_cap_gib"]),
        ) || error("response reached its declared resource cap after checkpointing")
    end
    if mode == "spin_up_addition_ph"
        removal = reconstruct_removal(addition)
    end
    defects = channel_defects(addition, removal)
    passed = mode == "all_eight" && maximum((
        defects.particle_hole, defects.spin, defects.equal_time, defects.inversion
    )) <= tolerance
    write_response_channels(joinpath(output, "raw_response_channels.csv"), t, addition, removal)
    write_green_matrix(joinpath(output, "retarded_green_matrix.csv"), t, defects.total)
    result = Dict(
        "model_digest" => digest,
        "U" => U,
        "epsilon" => -U / 2,
        "td" => td,
        "bath_spin_orbitals_total" => M,
        "bath_spinful_sites_per_impurity" => g.m,
        "electron_mps_sites_including_dimer" => g.L,
        "branch_mode" => mode,
        "requested_branch_mode" => requested_mode,
        "reduction_gate_validated" => reduction_gate_validated,
        "source_count" => length(states),
        "identity_gate_passed" => passed,
        "identity_gate_tolerance" => tolerance,
        "particle_hole_defect" => defects.particle_hole,
        "spin_defect" => defects.spin,
        "equal_time_defect" => defects.equal_time,
        "inversion_defect" => defects.inversion,
        "dt" => dt,
        "tmax" => tmax,
        "cutoff" => cutoff,
        "maxdim" => maxdim,
        "updater_krylov_dimension" => Int(tdvp_config["updater_krylov_dimension"]),
        "updater_tolerance" => Float64(tdvp_config["updater_tolerance"]),
        "neutral_maximum_discarded_weight" => neutral_maximum_discarded,
        "neutral_cumulative_discarded_weight" => neutral_cumulative_discarded,
        "neutral_actual_maximum_bond_dimension" => neutral_maximum_bond,
        "neutral_norm_squared_by_time" => neutral_norm,
        "neutral_maximum_norm_drift" => maximum(abs.(neutral_norm .- neutral_norm[1])),
        "neutral_energy_by_time" => neutral_energy,
        "neutral_maximum_energy_drift" => maximum(abs.(neutral_energy .- neutral_energy[1])),
        "source_maximum_discarded_weight" => maximum_discarded,
        "source_cumulative_discarded_weight" => cumulative_discarded,
        "source_actual_maximum_bond_dimension" => maximum_bond,
        "source_norm_squared_by_time" => source_norm,
        "neutral_wall_seconds" => neutral_wall,
        "source_wall_seconds" => wall_by_state,
        "wall_seconds" => time() - started,
        "peak_rss_bytes" => Sys.maxrss(),
        "ground_checkpoint" => relpath(ground_path, DIMER_ROOT),
        "checkpoint_directory" => relpath(checkpoint_directory, DIMER_ROOT),
    )
    write_toml(joinpath(output, "response.toml"), result)
    return result
end

"""One bounded U=0 DMRG and all-source two-site TDVP sign calibration."""
function calibrate_u0(config, output)
    model = config["model"]
    cal = config["calibration"]
    M = Int(cal["bath_spin_orbitals_total"])
    g = dimer_geometry(M)
    U = 0.0
    td = Float64(model["td"])
    W = Float64(model["W"])
    Gamma = Float64(model["Gamma"])
    seed = Int(cal["seed"])
    Random.seed!(seed)
    sites = siteinds("Electron", g.L; conserve_qns = true)
    H = build_dimer_hamiltonian(sites; M, U, td, W, Gamma)
    psi0 = random_mps(
        sites, dimer_initial_state(g.L); linkdims = Int(cal["initial_link_dimension"])
    )
    started = time()
    sweeps = Int(cal["dmrg_sweeps"])
    maxdim = Int(cal["maxdim"])
    cutoff = Float64(cal["cutoff"])
    energy, psi = dmrg(
        H, psi0;
        nsweeps = sweeps,
        maxdim = [min(maxdim, 8 * 2^(min(k - 1, 3))) for k in 1:sweeps],
        cutoff,
        noise = [k <= 2 ? 1.0e-7 : 0.0 for k in 1:sweeps],
        outputlevel = 0,
    )
    norm2 = real(inner(psi, psi))
    mean_energy = real(inner(psi', H, psi)) / norm2
    second_moment = real(inner(H, psi, H, psi)) / norm2
    sigma_E = sqrt(max(second_moment - mean_energy^2, 0.0))
    dt = Float64(cal["dt"])
    tmax = Float64(cal["tmax"])
    t = collect(0.0:dt:tmax)
    impurity = (g.d1, g.d2)
    spin_names = ("up", "down")
    states = Dict{Tuple{Symbol, Int, Int}, MPS}()
    for (sigma, spin) in enumerate(spin_names), j in 1:2
        create, destroy, _ = spin_operators(spin)
        states[(:addition, j, sigma)] = fermion_apply(psi, sites, impurity[j], create)
        states[(:removal, j, sigma)] = fermion_apply(psi, sites, impurity[j], destroy)
    end
    neutral = deepcopy(psi)
    add = zeros(ComplexF64, length(t), 2, 2, 2)
    rem = similar(add)
    max_discarded = Dict(string(key) => 0.0 for key in keys(states))
    cumulative_discarded = Dict(string(key) => 0.0 for key in keys(states))
    neutral_max_discarded = 0.0
    neutral_cumulative_discarded = 0.0
    stopped = false
    for k in eachindex(t)
        for sigma in 1:2, source in 1:2, sink in 1:2
            spin = spin_names[sigma]
            _, destroy, _ = spin_operators(spin)
            add[k, sink, source, sigma] = -im * fermion_matrix_element(
                neutral, states[(:addition, source, sigma)], sites, impurity[sink], destroy
            )
            rem[k, sink, source, sigma] = -im * fermion_matrix_element(
                states[(:removal, source, sigma)], neutral, sites, impurity[sink], destroy
            )
        end
        k == lastindex(t) && break
        neutral, trunc, cumulative, _, _, _ = tdvp_step(
            H, neutral;
            dt,
            cutoff,
            maxdim,
            nsite = 2,
            updater_krylovdim = Int(cal["updater_krylov_dimension"]),
            updater_tolerance = Float64(cal["updater_tolerance"]),
        )
        neutral_max_discarded = max(neutral_max_discarded, trunc)
        neutral_cumulative_discarded += cumulative
        for key in keys(states)
            states[key], trunc, cumulative, _, _, _ = tdvp_step(
                H, states[key];
                dt,
                cutoff,
                maxdim,
                nsite = 2,
                updater_krylovdim = Int(cal["updater_krylov_dimension"]),
                updater_tolerance = Float64(cal["updater_tolerance"]),
            )
            name = string(key)
            max_discarded[name] = max(max_discarded[name], trunc)
            cumulative_discarded[name] += cumulative
        end
        if !resource_available(
            started, Float64(cal["wall_cap_seconds"]), Float64(cal["rss_cap_gib"])
        )
            stopped = true
            open(joinpath(output, "calibration_resource_stop.jls"), "w") do io
                serialize(io, (; completed_time = t[k + 1], neutral, states))
            end
            break
        end
    end
    stopped && error("calibration reached its declared resource cap after checkpointing")
    G = dropdims(mean(add + rem; dims = 4); dims = 4)
    exact = exact_u0_green(t, M; td, W, Gamma)
    difference = vec(G - exact)
    relative = norm(difference) / norm(vec(exact))
    maximum_absolute = maximum(abs.(difference))
    write_green_matrix(joinpath(output, "calibration_TDVP_green_matrix.csv"), t, G)
    write_green_matrix(joinpath(output, "calibration_exact_green_matrix.csv"), t, exact)
    checkpoint = joinpath(output, "calibration_final_states.jls")
    open(checkpoint, "w") do io
        serialize(io, (; t, sites, neutral, states, add, rem, M, U, td, W, Gamma))
    end
    result = Dict(
        "scope" => "bounded U=0 sign, phase, and timing calibration; not production",
        "model_digest" => model_digest(; M, U, td, W, Gamma),
        "bath_spin_orbitals_total" => M,
        "bath_spinful_sites_per_impurity" => g.m,
        "electron_mps_sites_including_dimer" => g.L,
        "particle_sector" => Dict("N_up" => g.L ÷ 2, "N_down" => g.L ÷ 2),
        "energy" => Float64(real(energy)),
        "energy_standard_deviation" => sigma_E,
        "actual_ground_maximum_bond_dimension" => maxlinkdim(psi),
        "equal_time_maximum_error" => maximum(
            abs.(G[1, :, :] + im .* Matrix{ComplexF64}(I, 2, 2))
        ),
        "complex_relative_L2_error" => relative,
        "maximum_absolute_error" => maximum_absolute,
        "neutral_maximum_discarded_weight" => neutral_max_discarded,
        "neutral_cumulative_discarded_weight" => neutral_cumulative_discarded,
        "source_maximum_discarded_weight" => max_discarded,
        "source_cumulative_discarded_weight" => cumulative_discarded,
        "wall_seconds" => time() - started,
        "peak_rss_bytes" => Sys.maxrss(),
        "checkpoint" => relpath(checkpoint, DIMER_ROOT),
    )
    write_toml(joinpath(output, "calibration.toml"), result)
    return result
end

function dimer_main()
    BLAS.set_num_threads(1)
    ITensors.Strided.disable_threads()
    ITensors.disable_threaded_blocksparse()
    config_path = joinpath(@__DIR__, "..", "..", "parameters.json")
    config = JSON.parsefile(config_path)["tdvp"]["hubbard_dimer"]
    if !isempty(ARGS)
        config["run"]["task"] = ARGS[1]
    end
    task = String(config["run"]["task"])
    output = joinpath(DIMER_ROOT, String(config["run"]["output"]))
    mkpath(output)
    runtime = Dict(
        "julia_version" => string(VERSION),
        "is_linux" => Sys.islinux(),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "itensors_version" => string(Base.pkgversion(ITensors)),
        "itensormps_version" => string(Base.pkgversion(ITensorMPS)),
    )
    write_toml(joinpath(output, "runtime.toml"), runtime)
    println(runtime)
    if task == "bath"
        println(bath_preflight(config, output))
    elseif task == "u0"
        println(u0_gate(config, output))
    elseif task == "calibrate"
        println(calibrate_u0(config, output))
    elseif task == "ground"
        println(dimer_ground(config, output))
    elseif task == "response"
        println(dimer_response(config, output))
    else
        error("task must be bath, u0, calibrate, ground, or response")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    dimer_main()
end
