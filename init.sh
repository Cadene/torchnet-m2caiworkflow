touch README.md
mkdir data
touch data/.gitkeep
mkdir data/external
touch data/external/.gitkeep
mkdir data/interim
touch data/interim/.gitkeep
mkdir data/processed
touch data/processed/.gitkeep
mkdir data/raw
touch data/raw/.gitkeep
mkdir docs
touch docs/.gitkeep
mkdir models
touch models/.gitkeep
mkdir models/raw
touch models/raw/.gitkeep
mkdir models/processed
touch models/processed/.gitkeep
touch requirements.txt
mkdir src
echo "# MacOs indexing files" > .gitignore
echo ".DS_Store" >> .gitignore
echo "._.DS_Store" >> .gitignore
echo "# Exclude data" >> .gitignore
echo "data/raw/*" >> .gitignore
echo "data/processed/*" >> .gitignore
echo "data/features/*" >> .gitignore
echo "data/interim/*" >> .gitignore
echo "# Exclude models" >> .gitignore
echo "models/raw/*" >> .gitignore
echo "models/processed/*" >> .gitignore
echo "!.gitkeep" >> .gitignore
