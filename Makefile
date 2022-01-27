# Change TARGET below to produce .py versions of the given .ipynb's.
TARGET = 02_train.py

all: $(TARGET)

%.py: %.ipynb
	jupyter nbconvert --RegexRemovePreprocessor.patterns="['^#noexport']" --to script $<

clean:
	$(RM) $(TARGET)
