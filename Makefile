# ============================================================
# Makefile for Hierarchical Clustering with Confidence
# ============================================================
.PHONY: fig1_sim figure1 figure2 fig3_sim figure3 fig4_sim figure4 fig5_sim figure5 fig6_sim figure6 figure7 fig8_sim figure8 fig9_sim figure9


# Variables
PYTHON := $(CURDIR)/.venv/bin/python
PYTHONPATH := $(CURDIR)/src
SRC_DIR := src
RESULTS_DIR := results
SIM_DIR := simulations
RAW_DIR := $(RESULTS_DIR)/raw
FIG_DIR := $(RESULTS_DIR)/figures
FIG1_RAW := results/raw/fig1
FIG3_RAW := results/raw/fig3
FIG4_RAW := results/raw/fig4_es
FIG5_RAW := results/raw/fig5
FIG6_RAW := results/raw/penguins

# Figure 1: dendrograms and K-hat histogram
fig1_sim:
	@echo "Running simulations for Figure 1."
	for b in 0 1 2 3 4 5 6 7 8 9; do \
		BATCH_ID=$$b NUM_BATCHES=10 REPS_PER_BATCH=10 $(PYTHON) simulations/fig1_simulations.py; \
	done

figure1:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig1
	@echo "Complete. Figure 1 saved in $(FIG_DIR)"

# Figure 2: WCSS/TSS and ARI
figure2:
	@echo "Running simulations for Figure 2"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.fig2_simulations
	@echo "Simulation complete. Results saved in $(RAW_DIR)"
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig2
	@echo "Complete. Figure 2 saved in $(FIG_DIR)"

# Figure 3: Comparison of ECDF and Type I error across randomization levels
fig3_sim_cluster:
	# slurm for reference
	@echo "Submitting simulations to cluster."
	sbatch scripts/run_typeI_r.sh
	sbatch scripts/run_validity_r.sh
	@echo "Submitted."

fig3_sim:
	# Here we use num_trials = 20 num_repeats = 1 for demonstration
	# In experiments we used 2000 trials for ECDF, 200 trials and 100 repeats for Type I error
	# Please change the parameters below accordingly to reproduce the plot.
	@echo "Running simulations for Figure 3"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.check_validity --K 2 --num_trials 20
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.check_validity --K 3 --num_trials 20
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.type1_error --K 2 --num_trials 20 --num_repeats 1
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.type1_error --K 3 --num_trials 20 --num_repeats 1
	@echo "Simulation completed."
figure3:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig3
	@echo "Complete. Figure 3 saved in $(FIG_DIR)"

# Figure 4: Power comparison
fig4_sim_cluster:
	# slurm for reference
	@echo "Submitting simulations to cluster."
	sbatch scripts/run_power_barber.sh
	sbatch scripts/run_power_gao.sh
	sbatch scripts/run_power_es_rand.sh
	@echo "Submitted."


TRIALS := 2
fig4_sim:
	# Here we use num_trials = 2 for demonstration
	# In experiments we used 2000 trials
	# Please change the parameters below accordingly to reproduce the plot.
	@echo "Running simulations for Figure 4."
	for K in 2 3; do \
		for L in single average complete; do \
		  	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.rand_power_es --K $$K --num_trials $(TRIALS) --linkage $$L; \
			PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.gao_power_es --linkage $$L --K $$K --num_trials $(TRIALS); \
			PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.barber_power_es --linkage $$L --K $$K --num_trials $(TRIALS); \
		done; \
	done

	for K in 2 3; do \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.rand_power_es --K $$K --num_trials $(TRIALS) --linkage minimax; \
	done
	@echo "Simulation completed."


figure4:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig4
	@echo "Complete. Figure 4 saved in $(FIG_DIR)"


# Figure 5: Histogram of Khat with varying delta
fig5_sim_cluster:
	# slurm for reference
	@echo "Submitting simulations to cluster."
	sbatch scripts/run_chooseK_batch.sh
	@echo "Submitted."


DELTA_LIST := 4 8 12
N_REP := 5
fig5_sim:
	# Here we use num_reps = 5 and delta = 4,8,12 for demonstration
	# In experiments we used 100 trials and delta = 4,6,8,10,12,14
	# Please change the parameters below accordingly to reproduce the plot.
	@echo "Running simulations for Figure 5."
	@for d in $(DELTA_LIST); do \
		for t in $$(seq 0 $(shell expr $(N_REP) - 1)); do \
			echo "delta=$$d trial=$$t"; \
			PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.Khat_delta --delta $$d --trial $$t; \
		done; \
	done

figure5:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig5
	@echo "Complete. Figure 5 saved in $(FIG_DIR)"

# Figure 6: Penguin Dataset
fig6_sim_cluster:
	# slurm for reference
	@echo "Submitting simulations to cluster."
	sbatch scripts/run_penguin_batch.sh
	@echo "Submitted."
	@echo "Running stability experiments..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.stability
	@echo "Complete."

N_TRIALS := 1
fig6_sim:
	@echo "Running simulations for Figure 6..."
	@for t in $$(seq 0 $(shell expr $(N_TRIALS) - 1)); do \
		echo "Running trial $$t"; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.penguin_K_batch --trial_id $$t; \
	done
	@echo "Running stability experiments..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.stability
	@echo "Complete."

figure6:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig6
	@echo "Complete. Figure 6 saved in $(FIG_DIR)"


# Figure 7: WCSS/TSS and ARI across different deltas
figure7:
	@echo "Running simulations for Figure 7"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.fig7_simulations
	@echo "Simulation complete. Results saved in $(RAW_DIR)"
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig7
	@echo "Complete. Figure 7 saved in $(FIG_DIR)"

# Figure 8: ECDF with different linkages
fig8_sim_cluster:
	@echo "Submitting simulations to cluster."
	sbatch scripts/run_fig8sim.sh
	@echo "Submitted."

fig8_sim:
	# Here we use num_trails = 20 for demonstration
	# In experiments we used 2000 trials.
	# Please change the parameters below accordingly to reproduce the plot.
	@echo "Running simulations for Figure 8..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.fig8_simulations --K 2 --num_trials 20
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.fig8_simulations --K 3 --num_trials 20
	@echo "Complete."


figure8:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig8
	@echo "Complete. Figure 8 saved in $(FIG_DIR)"

# Figure 9: Histogram for estimated K across different true K
fig9_sim_cluster:
	@echo "Submitting simulations to cluster."
	sbatch scripts/run_chooseK_varyK.sh
	@echo "Submitted."


N := 30
P := 2
DELTA := 6
K_LIST := 2 3 4
N_REP := 3

fig9_sim:
	@for K in $(K_LIST); do \
		for t in $$(seq 0 $(shell expr $(N_REP) - 1)); do \
			echo "K=$$K trial=$$t (n=$(N), p=$(P), delta=$(DELTA))"; \
			PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m simulations.fig9_sim --K $$K --n $(N) --p $(P) --delta $(DELTA) --trial $$t; \
		done; \
	done

figure9:
	cd plotting && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m plot_fig9
	@echo "Complete. Figure 9 saved in $(FIG_DIR)"