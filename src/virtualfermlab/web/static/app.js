/* VirtualFermLab Web UI JavaScript */

// Helper: render a Plotly chart from JSON {data, layout} into a container
function renderPlotly(containerId, plotJson) {
    var el = document.getElementById(containerId);
    if (!el) return;
    if (!plotJson || !plotJson.data) {
        el.innerHTML = '<p class="text-muted">No chart data available.</p>';
        return;
    }
    Plotly.newPlot(el, plotJson.data, plotJson.layout, {responsive: true});
}

// Global state
var currentStep = 1;
var strainProfile = null;
var currentTaskId = null;
var currentRunMode = 'single';
var pollInterval = null;

// ============================================================
// Wizard navigation
// ============================================================

function goToStep(step) {
    // Hide all panels
    var panels = document.querySelectorAll('.wizard-panel');
    for (var i = 0; i < panels.length; i++) {
        panels[i].classList.add('d-none');
    }
    // Show target
    document.getElementById('step-' + step).classList.remove('d-none');

    // Update indicators
    for (var j = 1; j <= 5; j++) {
        var ind = document.getElementById('ind-' + j);
        ind.classList.remove('active');
        if (j < step) {
            ind.classList.add('completed');
        } else {
            ind.classList.remove('completed');
        }
    }
    document.getElementById('ind-' + step).classList.add('active');
    currentStep = step;
}

// ============================================================
// Step 1: Strain loading
// ============================================================

function loadStrains() {
    fetch('/api/strains')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var sel = document.getElementById('strain-select');
            sel.innerHTML = '<option value="">-- Select Strain --</option>';
            data.strains.forEach(function(s) {
                var opt = document.createElement('option');
                opt.value = s;
                opt.textContent = s;
                sel.appendChild(opt);
            });
        });
}

document.addEventListener('DOMContentLoaded', function() {
    loadStrains();

    document.getElementById('strain-select').addEventListener('change', function() {
        var name = this.value;
        if (!name) return;
        fetch('/api/strain/' + name)
            .then(function(r) { return r.json(); })
            .then(function(profile) {
                strainProfile = profile;
                displayStrainSummary(profile);
                populateSubstrateDropdowns(profile);
                prefillPHParams(profile);
                document.getElementById('btn-step1-next').disabled = false;
            });
    });
});

function displayStrainSummary(profile) {
    var container = document.getElementById('strain-summary');
    var html = '<h6>' + profile.name + '</h6>';

    // Substrates
    var subs = profile.substrates || {};
    for (var key in subs) {
        var sub = subs[key];
        html += '<div class="param-card">';
        html += '<h6>' + sub.name + '</h6>';
        html += '<div class="param-row"><span class="param-name">mu_max</span><span class="param-value">' + sub.mu_max.value.toFixed(4) + ' (' + sub.mu_max.type + ')</span></div>';
        html += '<div class="param-row"><span class="param-name">Ks</span><span class="param-value">' + sub.Ks.value.toFixed(4) + ' (' + sub.Ks.type + ')</span></div>';
        html += '<div class="param-row"><span class="param-name">Yxs</span><span class="param-value">' + sub.Yxs.value.toFixed(4) + ' (' + sub.Yxs.type + ')</span></div>';
        html += '</div>';
    }

    // Cardinal pH & Inhibitions — collapsible "More" section
    var extraHtml = '';

    if (profile.cardinal_pH) {
        extraHtml += '<div class="param-card">';
        extraHtml += '<h6>Cardinal pH</h6>';
        extraHtml += '<div class="param-row"><span class="param-name">pH_min</span><span class="param-value">' + profile.cardinal_pH.pH_min.value.toFixed(1) + '</span></div>';
        extraHtml += '<div class="param-row"><span class="param-name">pH_opt</span><span class="param-value">' + profile.cardinal_pH.pH_opt.value.toFixed(1) + '</span></div>';
        extraHtml += '<div class="param-row"><span class="param-name">pH_max</span><span class="param-value">' + profile.cardinal_pH.pH_max.value.toFixed(1) + '</span></div>';
        extraHtml += '</div>';
    }

    if (profile.inhibitions && profile.inhibitions.length > 0) {
        extraHtml += '<div class="param-card">';
        extraHtml += '<h6>Inhibition</h6>';
        profile.inhibitions.forEach(function(inh) {
            extraHtml += '<div class="param-row"><span class="param-name">K_I (' + inh.inhibitor + ' on ' + inh.inhibited + ')</span><span class="param-value">' + inh.K_I.value.toFixed(3) + '</span></div>';
        });
        extraHtml += '</div>';
    }

    if (extraHtml) {
        html += '<div id="strain-extra" style="display:none;">' + extraHtml + '</div>';
        html += '<button type="button" class="btn btn-sm btn-outline-secondary mt-1" id="btn-strain-more" onclick="toggleStrainExtra()">Show more</button>';
    }

    container.innerHTML = html;
}

function toggleStrainExtra() {
    var extra = document.getElementById('strain-extra');
    var btn = document.getElementById('btn-strain-more');
    if (extra.style.display === 'none') {
        extra.style.display = '';
        btn.textContent = 'Show less';
    } else {
        extra.style.display = 'none';
        btn.textContent = 'Show more';
    }
}

function populateSubstrateDropdowns(profile) {
    var selA = document.getElementById('substrate-a');
    var selB = document.getElementById('substrate-b');
    selA.innerHTML = '';
    selB.innerHTML = '';
    selB.innerHTML = '<option value="">None</option>';

    var subs = profile.substrates || {};
    for (var key in subs) {
        var optA = document.createElement('option');
        optA.value = key;
        optA.textContent = key;
        selA.appendChild(optA);

        var optB = document.createElement('option');
        optB.value = key;
        optB.textContent = key;
        selB.appendChild(optB);
    }
}

function prefillPHParams(profile) {
    if (profile.cardinal_pH) {
        document.getElementById('pH_min').value = profile.cardinal_pH.pH_min.value;
        document.getElementById('pH_opt').value = profile.cardinal_pH.pH_opt.value;
        document.getElementById('pH_max').value = profile.cardinal_pH.pH_max.value;
    }
}

// ============================================================
// Step 2: Toggle conditional fields & cross-step dependencies
// ============================================================

function togglePHFields() {
    var checked = document.getElementById('use-cardinal-ph').checked;
    document.getElementById('ph-fields').classList.toggle('d-none', !checked);
}

function toggleLagField() {
    var checked = document.getElementById('use-lag').checked;
    document.getElementById('lag-field').classList.toggle('d-none', !checked);
}

// N Substrates ↔ Step 3 (Substrate B, Ratio)
function syncSubstrateFields() {
    var nSub = parseInt(document.getElementById('n-substrates').value);
    var subBWrapper = document.getElementById('substrate-b-wrapper');
    var ratioWrapper = document.getElementById('ratio-wrapper');

    if (nSub === 1) {
        // Hide Substrate B and Ratio — force ratio=1 (all substrate A)
        subBWrapper.classList.add('d-none');
        ratioWrapper.classList.add('d-none');
        document.getElementById('ratio-input').value = '1.0';
        document.getElementById('ratio-slider').value = '1.0';
        document.getElementById('ratio-display').textContent = '1.00';
    } else {
        subBWrapper.classList.remove('d-none');
        ratioWrapper.classList.remove('d-none');
    }
}

// Enzyme mode ↔ Strain profile compatibility
function checkEnzymeCompat() {
    var mode = document.querySelector('input[name="enzyme_mode"]:checked').value;
    var warnEl = document.getElementById('enzyme-warning');

    if (mode !== 'direct' && strainProfile && !strainProfile.enzyme_params) {
        warnEl.textContent = 'Selected strain has no enzyme parameters. Default values will be used.';
        warnEl.classList.remove('d-none');
    } else {
        warnEl.classList.add('d-none');
    }
}

// ============================================================
// Validation before run
// ============================================================

function validateBeforeRun() {
    var warnings = [];
    var nSub = parseInt(document.getElementById('n-substrates').value);
    var subA = document.getElementById('substrate-a').value;
    var subB = document.getElementById('substrate-b').value;

    // N Substrates = 2 but same substrate selected
    if (nSub === 2 && subA && subB && subA === subB) {
        warnings.push('Substrate A and B are the same (' + subA + '). Select different substrates or set N Substrates = 1.');
    }

    // N Substrates = 2 but Substrate B not selected
    if (nSub === 2 && !subB) {
        warnings.push('N Substrates = 2 but Substrate B is not selected.');
    }

    // Cardinal pH: validate pH_min < pH_opt < pH_max
    var usePH = document.getElementById('use-cardinal-ph').checked;
    if (usePH) {
        var pMin = parseFloat(document.getElementById('pH_min').value);
        var pOpt = parseFloat(document.getElementById('pH_opt').value);
        var pMax = parseFloat(document.getElementById('pH_max').value);
        if (pMin >= pOpt || pOpt >= pMax) {
            warnings.push('Cardinal pH requires pH_min < pH_opt < pH_max (got ' + pMin + ', ' + pOpt + ', ' + pMax + ').');
        }
        var pH = parseFloat(document.getElementById('pH-input').value);
        if (pH < pMin || pH > pMax) {
            warnings.push('Experiment pH (' + pH + ') is outside cardinal range [' + pMin + ', ' + pMax + ']. Growth factor will be 0.');
        }
    }

    // Time validation
    var tStart = parseFloat(document.getElementById('time-start').value);
    var tEnd = parseFloat(document.getElementById('time-end').value);
    if (tEnd <= tStart) {
        warnings.push('Time End must be greater than Time Start.');
    }

    // Show or hide warnings
    var warnContainer = document.getElementById('run-warnings');
    var warnText = document.getElementById('run-warnings-text');
    if (warnings.length > 0) {
        warnText.innerHTML = warnings.join('<br>');
        warnContainer.classList.remove('d-none');
        return false;
    } else {
        warnContainer.classList.add('d-none');
        return true;
    }
}

// ============================================================
// Step 4: Mode switching
// ============================================================

function toggleRunMode() {
    var mode = document.querySelector('input[name="run_mode"]:checked').value;
    currentRunMode = mode;

    document.getElementById('mc-settings').classList.toggle('d-none', mode === 'single');
    document.getElementById('ve-settings').classList.toggle('d-none', mode !== 'virtual_experiment');
    document.getElementById('method-wrapper').classList.toggle('d-none', mode !== 'single');

    var btnText = {
        'single': 'Run Simulation',
        'monte_carlo': 'Run Monte Carlo',
        'virtual_experiment': 'Run Virtual Experiment'
    };
    document.getElementById('btn-run').textContent = btnText[mode];
}

// ============================================================
// Collect form data
// ============================================================

function collectFormData() {
    var strain = document.getElementById('strain-select').value;
    var usePH = document.getElementById('use-cardinal-ph').checked;
    var useLag = document.getElementById('use-lag').checked;
    var nSub = parseInt(document.getElementById('n-substrates').value);

    var data = {
        strain: strain,
        growth_model: document.querySelector('input[name="growth_model"]:checked').value,
        enzyme_mode: document.querySelector('input[name="enzyme_mode"]:checked').value,
        n_substrates: nSub,
        n_feeds: 1,
        use_cardinal_pH: usePH,
        use_lag: useLag,
        substrate_A: document.getElementById('substrate-a').value,
        substrate_B: nSub === 2 ? (document.getElementById('substrate-b').value || null) : null,
        ratio: nSub === 2 ? parseFloat(document.getElementById('ratio-input').value) : 1.0,
        pH: parseFloat(document.getElementById('pH-input').value),
        total_concentration: parseFloat(document.getElementById('total-conc').value),
        dilution_rate: parseFloat(document.getElementById('dilution-rate').value),
        X0: parseFloat(document.getElementById('X0-input').value),
        time_start: parseFloat(document.getElementById('time-start').value),
        time_end: parseFloat(document.getElementById('time-end').value),
        n_points: parseInt(document.getElementById('n-points').value),
        method: document.getElementById('method-select').value
    };

    if (usePH) {
        data.pH_min = parseFloat(document.getElementById('pH_min').value);
        data.pH_opt = parseFloat(document.getElementById('pH_opt').value);
        data.pH_max = parseFloat(document.getElementById('pH_max').value);
    }
    if (useLag) {
        data.lag = parseFloat(document.getElementById('lag').value);
    }

    return data;
}

// ============================================================
// Parameters summary
// ============================================================

function displayParamsSummary(data) {
    var strainTb = document.getElementById('params-strain-table');
    var condTb = document.getElementById('params-condition-table');
    var simTb = document.getElementById('params-sim-table');
    strainTb.innerHTML = '';
    condTb.innerHTML = '';
    simTb.innerHTML = '';

    // Strain & Model
    addSummaryRow(strainTb, 'Strain', data.strain);
    addSummaryRow(strainTb, 'Growth Model', data.growth_model);
    addSummaryRow(strainTb, 'Enzyme Mode', data.enzyme_mode);
    addSummaryRow(strainTb, 'N Substrates', data.n_substrates);
    addSummaryRow(strainTb, 'Cardinal pH', data.use_cardinal_pH ? 'Yes' : 'No');
    if (data.use_cardinal_pH) {
        addSummaryRow(strainTb, 'pH_min', data.pH_min);
        addSummaryRow(strainTb, 'pH_opt', data.pH_opt);
        addSummaryRow(strainTb, 'pH_max', data.pH_max);
    }
    addSummaryRow(strainTb, 'Lag Phase', data.use_lag ? 'Yes' : 'No');
    if (data.use_lag) {
        addSummaryRow(strainTb, 'Lag Duration', data.lag + ' h');
    }

    // Condition
    addSummaryRow(condTb, 'Substrate A', data.substrate_A);
    if (data.n_substrates === 2) {
        addSummaryRow(condTb, 'Substrate B', data.substrate_B || '-');
        addSummaryRow(condTb, 'Ratio (A fraction)', data.ratio);
    }
    addSummaryRow(condTb, 'pH', data.pH);
    addSummaryRow(condTb, 'Total Conc.', data.total_concentration + ' g/L');
    addSummaryRow(condTb, 'Dilution Rate', data.dilution_rate + ' 1/h');
    addSummaryRow(condTb, 'Initial Biomass (X0)', data.X0 + ' g/L');

    // Simulation settings
    var modeLabels = {
        'single': 'Single Simulation',
        'monte_carlo': 'Monte Carlo',
        'virtual_experiment': 'Virtual Experiment'
    };
    addSummaryRow(simTb, 'Mode', modeLabels[currentRunMode]);
    addSummaryRow(simTb, 'Time Range', data.time_start + ' - ' + data.time_end + ' h');
    addSummaryRow(simTb, 'N Points', data.n_points);
    if (currentRunMode === 'single') {
        addSummaryRow(simTb, 'Method', data.method);
    }
    if (currentRunMode === 'monte_carlo') {
        addSummaryRow(simTb, 'N Samples', data.n_samples);
        addSummaryRow(simTb, 'Seed', data.seed);
    }
    if (currentRunMode === 'virtual_experiment') {
        addSummaryRow(simTb, 'pH Range', data.pH_range_min + ' - ' + data.pH_range_max);
        addSummaryRow(simTb, 'Ratio Range', data.ratio_range_min + ' - ' + data.ratio_range_max);
        addSummaryRow(simTb, 'N Continuous', data.n_continuous);
        addSummaryRow(simTb, 'MC Samples/Cond', data.n_samples);
        addSummaryRow(simTb, 'Seed', data.seed);
        addSummaryRow(simTb, 'Ranking Score', data.obj_ranking);
        addSummaryRow(simTb, 'Heatmap Metric', data.obj_heatmap);
        addSummaryRow(simTb, 'Pareto Objectives', data.obj_pareto_1 + ' vs ' + data.obj_pareto_2);
    }

    document.getElementById('params-summary').classList.remove('d-none');
}

// ============================================================
// Run simulation
// ============================================================

function runSimulation() {
    if (!validateBeforeRun()) return;

    var data = collectFormData();
    var btn = document.getElementById('btn-run');
    btn.disabled = true;
    btn.textContent = 'Running...';

    if (currentRunMode === 'monte_carlo') {
        data.n_samples = parseInt(document.getElementById('n-samples').value);
        data.seed = parseInt(document.getElementById('mc-seed').value);
    } else if (currentRunMode === 'virtual_experiment') {
        data.pH_range_min = parseFloat(document.getElementById('pH-range-min').value);
        data.pH_range_max = parseFloat(document.getElementById('pH-range-max').value);
        data.ratio_range_min = parseFloat(document.getElementById('ratio-range-min').value);
        data.ratio_range_max = parseFloat(document.getElementById('ratio-range-max').value);
        data.n_continuous = parseInt(document.getElementById('n-continuous').value);
        data.n_samples = parseInt(document.getElementById('ve-n-samples').value);
        data.seed = parseInt(document.getElementById('mc-seed').value || '42');
        data.obj_ranking = document.getElementById('obj-ranking').value;
        data.obj_heatmap = document.getElementById('obj-heatmap').value;
        data.obj_pareto_1 = document.getElementById('obj-pareto-1').value;
        data.obj_pareto_2 = document.getElementById('obj-pareto-2').value;
    }

    // Show parameter summary before running
    displayParamsSummary(data);

    if (currentRunMode === 'single') {
        runSingle(data);
    } else if (currentRunMode === 'monte_carlo') {
        runMonteCarlo(data);
    } else {
        runVirtualExperiment(data);
    }
}

function resetRunButton() {
    var btn = document.getElementById('btn-run');
    btn.disabled = false;
    var btnText = {
        'single': 'Run Simulation',
        'monte_carlo': 'Run Monte Carlo',
        'virtual_experiment': 'Run Virtual Experiment'
    };
    btn.textContent = btnText[currentRunMode];
}

// ============================================================
// Single simulation
// ============================================================

function runSingle(data) {
    fetch('/api/simulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(function(r) { return r.json(); })
    .then(function(result) {
        if (result.error) {
            alert('Error: ' + result.error);
            resetRunButton();
            return;
        }

        currentTaskId = result.task_id;

        // Show results
        document.getElementById('single-result').classList.remove('d-none');
        document.getElementById('mc-result').classList.add('d-none');
        document.getElementById('ve-result').classList.add('d-none');

        // Render trajectory chart via Plotly
        renderPlotly('trajectory-chart', result.plot_json);

        // Summary table
        var tbody = document.getElementById('summary-table');
        tbody.innerHTML = '';
        var summary = result.summary;
        addSummaryRow(tbody, 'Y_X/S (Yield)', summary.yield_biomass.toFixed(4) + ' g/g');
        addSummaryRow(tbody, 'Final Biomass', summary.final_biomass.toFixed(4) + ' g/L');
        addSummaryRow(tbody, '\u03BC_max (observed)', summary.mu_max_effective.toFixed(4) + ' 1/h');
        for (var key in summary.final_substrates) {
            addSummaryRow(tbody, key + ' (final)', summary.final_substrates[key].toFixed(4) + ' g/L');
        }

        document.getElementById('export-buttons').style.display = '';
        goToStep(5);
        resetRunButton();
    })
    .catch(function(err) {
        alert('Request failed: ' + err);
        resetRunButton();
    });
}

function addSummaryRow(tbody, label, value) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td class="fw-bold">' + label + '</td><td>' + value + '</td>';
    tbody.appendChild(tr);

    // Populate v2 gradient cards for single simulation
    if (tbody.id === 'summary-table') {
        if (label.indexOf('Y_X/S') >= 0 || label.indexOf('Yield') >= 0) {
            var el = document.getElementById('card-yield');
            if (el) el.textContent = value;
        } else if (label.indexOf('Final Biomass') >= 0) {
            var el = document.getElementById('card-biomass');
            if (el) el.textContent = value;
        } else if (label.indexOf('\u03BC_max') >= 0 || label.indexOf('mu_max') >= 0) {
            var el = document.getElementById('card-mu');
            if (el) el.textContent = value;
        }
    }
    // Populate v2 gradient cards for Monte Carlo
    if (tbody.id === 'mc-summary-table') {
        if (label === 'Mean Y_X/S') {
            var el = document.getElementById('mc-card-yield');
            if (el) el.textContent = value;
        } else if (label.indexOf('Mean \u03BC_max') >= 0 || label === 'Mean mu_max') {
            var el = document.getElementById('mc-card-mu');
            if (el) el.textContent = value;
        } else if (label === 'Std Y_X/S') {
            var el = document.getElementById('mc-card-std-yield');
            if (el) el.textContent = value;
        } else if (label === 'Valid Runs') {
            var el = document.getElementById('mc-card-valid');
            if (el) el.textContent = value;
        }
    }
}

// ============================================================
// Monte Carlo
// ============================================================

function runMonteCarlo(data) {
    fetch('/api/monte-carlo/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(function(r) { return r.json(); })
    .then(function(result) {
        if (result.error) {
            alert('Error: ' + result.error);
            resetRunButton();
            return;
        }
        currentTaskId = result.task_id;
        showProgressBar();
        pollProgress('/api/monte-carlo/status/', '/api/monte-carlo/result/');
    })
    .catch(function(err) {
        alert('Request failed: ' + err);
        resetRunButton();
    });
}

function showProgressBar() {
    document.getElementById('progress-container').classList.remove('d-none');
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-bar').textContent = '0%';
    document.getElementById('progress-text').textContent = 'Starting...';
}

function hideProgressBar() {
    document.getElementById('progress-container').classList.add('d-none');
}

function pollProgress(statusUrl, resultUrl) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(function() {
        fetch(statusUrl + currentTaskId)
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.status === 'running') {
                    var pct = Math.round(data.progress / data.total * 100);
                    document.getElementById('progress-bar').style.width = pct + '%';
                    document.getElementById('progress-bar').textContent = pct + '%';
                    document.getElementById('progress-text').textContent =
                        'Progress: ' + data.progress + ' / ' + data.total;
                } else if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    document.getElementById('progress-bar').style.width = '100%';
                    document.getElementById('progress-bar').textContent = '100%';
                    document.getElementById('progress-text').textContent = 'Complete!';

                    // Fetch results
                    fetch(resultUrl + currentTaskId)
                        .then(function(r) { return r.json(); })
                        .then(function(result) {
                            hideProgressBar();
                            if (resultUrl.indexOf('monte-carlo') >= 0) {
                                displayMCResult(result);
                            } else {
                                displayVEResult(result);
                            }
                            resetRunButton();
                        });
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    hideProgressBar();
                    alert('Task failed: ' + (data.error || 'Unknown error'));
                    resetRunButton();
                }
            });
    }, 2000);
}

function displayMCResult(result) {
    document.getElementById('single-result').classList.add('d-none');
    document.getElementById('mc-result').classList.remove('d-none');
    document.getElementById('ve-result').classList.add('d-none');

    // Render charts via Plotly
    renderPlotly('mc-envelope-chart', result.envelope_json);
    renderPlotly('mc-dist-chart', result.dist_json);

    var tbody = document.getElementById('mc-summary-table');
    tbody.innerHTML = '';
    var s = result.summary;
    addSummaryRow(tbody, 'Mean Y_X/S', s.mean_yield.toFixed(4) + ' g/g');
    addSummaryRow(tbody, 'Std Y_X/S', s.std_yield.toFixed(4));
    addSummaryRow(tbody, 'Mean \u03BC_max', s.mean_mu_max.toFixed(4) + ' 1/h');
    addSummaryRow(tbody, 'Std \u03BC_max', s.std_mu_max.toFixed(4));
    addSummaryRow(tbody, '95% CI (Y_X/S)', '[' + s.ci95_yield[0].toFixed(4) + ', ' + s.ci95_yield[1].toFixed(4) + ']');
    addSummaryRow(tbody, 'N Samples', s.n_samples);
    addSummaryRow(tbody, 'Valid Runs', s.n_valid + ' / ' + s.n_samples);

    document.getElementById('export-buttons').style.display = '';
    goToStep(5);
}

// ============================================================
// Virtual Experiment
// ============================================================

function runVirtualExperiment(data) {
    fetch('/api/virtual-experiment/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(function(r) { return r.json(); })
    .then(function(result) {
        if (result.error) {
            alert('Error: ' + result.error);
            resetRunButton();
            return;
        }
        currentTaskId = result.task_id;
        showProgressBar();
        document.getElementById('progress-text').textContent =
            'Running ' + result.n_conditions + ' conditions...';
        pollProgress('/api/virtual-experiment/status/', '/api/virtual-experiment/result/');
    })
    .catch(function(err) {
        alert('Request failed: ' + err);
        resetRunButton();
    });
}

function displayVEResult(result) {
    document.getElementById('single-result').classList.add('d-none');
    document.getElementById('mc-result').classList.add('d-none');
    document.getElementById('ve-result').classList.remove('d-none');

    // Rankings is still HTML table
    document.getElementById('rankings-container').innerHTML = result.rankings_html;

    // Heatmap and Pareto via Plotly
    if (result.heatmap_json) {
        renderPlotly('heatmap-chart', result.heatmap_json);
    } else {
        document.getElementById('heatmap-chart').innerHTML = '<p class="text-muted">Not enough data for heatmap.</p>';
    }

    if (result.pareto_json) {
        renderPlotly('pareto-chart', result.pareto_json);
    } else {
        document.getElementById('pareto-chart').innerHTML = '<p class="text-muted">Not enough data for Pareto front.</p>';
    }

    document.getElementById('export-buttons').style.display = '';
    goToStep(5);
}

// ============================================================
// Export
// ============================================================

function exportCSV() {
    if (currentTaskId) {
        window.location.href = '/api/export/csv/' + currentTaskId;
    }
}

function exportJSON() {
    if (currentTaskId) {
        window.location.href = '/api/export/json/' + currentTaskId;
    }
}

// ============================================================
// Strain Discovery
// ============================================================

var discoveryTaskId = null;
var discoveryPollInterval = null;

var _stageIds = {
    search: 'disc-stage-search',
    extract: 'disc-stage-extract',
    store: 'disc-stage-store',
    taxonomy: 'disc-stage-taxonomy',
    build_profile: 'disc-stage-build'
};

var _stageOrder = ['search', 'extract', 'store', 'taxonomy', 'build_profile'];

function startDiscovery() {
    var strainName = document.getElementById('strain-input').value.trim();
    if (!strainName) {
        alert('Please enter a strain name.');
        return;
    }

    // Reset UI
    var panel = document.getElementById('discovery-panel');
    panel.classList.remove('d-none');
    document.getElementById('discovery-result').classList.add('d-none');
    document.getElementById('btn-discover').disabled = true;
    document.getElementById('btn-discover').textContent = 'Searching...';

    // Reset all stage icons
    _stageOrder.forEach(function(stage) {
        var el = document.getElementById(_stageIds[stage]);
        el.className = 'discovery-stage';
        el.querySelector('.stage-icon').innerHTML = '&#9711;';
    });

    fetch('/api/discovery/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({strain_name: strainName})
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            alert('Error: ' + data.error);
            _resetDiscoveryButton();
            return;
        }
        discoveryTaskId = data.task_id;
        _pollDiscovery();
    })
    .catch(function(err) {
        alert('Discovery request failed: ' + err);
        _resetDiscoveryButton();
    });
}

function _resetDiscoveryButton() {
    document.getElementById('btn-discover').disabled = false;
    document.getElementById('btn-discover').textContent = 'Discover';
}

function _pollDiscovery() {
    if (discoveryPollInterval) clearInterval(discoveryPollInterval);

    discoveryPollInterval = setInterval(function() {
        fetch('/api/discovery/status/' + discoveryTaskId)
            .then(function(r) { return r.json(); })
            .then(function(data) {
                // Update stage icons based on progress
                var currentProgress = data.progress || 0;
                _stageOrder.forEach(function(stage, idx) {
                    var el = document.getElementById(_stageIds[stage]);
                    var stageNum = idx + 1;
                    if (stageNum < currentProgress) {
                        el.className = 'discovery-stage stage-completed';
                        el.querySelector('.stage-icon').innerHTML = '&#10003;';
                    } else if (stageNum === currentProgress) {
                        el.className = 'discovery-stage stage-running';
                        el.querySelector('.stage-icon').innerHTML = '&#9881;';
                    }
                });

                if (data.status === 'completed') {
                    clearInterval(discoveryPollInterval);
                    discoveryPollInterval = null;
                    // Mark all completed
                    _stageOrder.forEach(function(stage) {
                        var el = document.getElementById(_stageIds[stage]);
                        el.className = 'discovery-stage stage-completed';
                        el.querySelector('.stage-icon').innerHTML = '&#10003;';
                    });
                    _fetchDiscoveryResult();
                } else if (data.status === 'failed') {
                    clearInterval(discoveryPollInterval);
                    discoveryPollInterval = null;
                    _resetDiscoveryButton();
                    alert('Discovery failed: ' + (data.error || 'Unknown error'));
                }
            });
    }, 2000);
}

function _fetchDiscoveryResult() {
    fetch('/api/discovery/result/' + discoveryTaskId)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            _resetDiscoveryButton();

            // Show summary
            var resultDiv = document.getElementById('discovery-result');
            resultDiv.classList.remove('d-none');
            var html = '<strong>Source:</strong> ' + data.source;
            html += ' | <strong>Papers:</strong> ' + data.papers_found;
            html += ' | <strong>Params extracted:</strong> ' + data.params_extracted;
            if (data.similar_strain) {
                html += '<br><strong>Most similar strain:</strong> ' + data.similar_strain;
                html += ' (similarity: ' + (data.similarity_score != null ? (data.similarity_score * 100).toFixed(0) + '%' : '?') + ')';
            }
            resultDiv.innerHTML = html;

            // Load the profile into the UI
            if (data.profile) {
                strainProfile = data.profile;
                displayStrainSummary(data.profile);
                populateSubstrateDropdowns(data.profile);
                prefillPHParams(data.profile);
                document.getElementById('btn-step1-next').disabled = false;

                // Set the dropdown to show the discovered strain name
                var sel = document.getElementById('strain-select');
                // Check if option already exists
                var found = false;
                for (var i = 0; i < sel.options.length; i++) {
                    if (sel.options[i].value === data.strain_name) {
                        sel.value = data.strain_name;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    var opt = document.createElement('option');
                    opt.value = data.strain_name;
                    opt.textContent = data.strain_name + ' (discovered)';
                    sel.appendChild(opt);
                    sel.value = data.strain_name;
                }
            }
        })
        .catch(function(err) {
            _resetDiscoveryButton();
            alert('Failed to fetch discovery result: ' + err);
        });
}
