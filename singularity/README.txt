# Create the overlay

```bash
sbatch --output=install.log --cpus-per-task=4 --mem=20GB --time=00:30:00 --wrap='bash create-overlay.sh'
```

# Run the script

```bash
sbatch <(bash gen-batch.sh "python main.py")
```
