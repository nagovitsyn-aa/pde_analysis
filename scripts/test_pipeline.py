from pde_analysis.pipeline.add_run import add_run_from_h5

run_id = add_run_from_h5(
    experiment_name="test_pipeline",
    h5_path="data/h5/RK_Sun-W14_dx=0.1_Λ=0p1_u=1p.h5"
)

print("Created run:", run_id)