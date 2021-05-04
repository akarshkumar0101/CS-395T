#Given a DataSequence, return a dictionary whose keys are the fMRI timestamps
#and whose values are the words spoken between the last timestamp and this 
#timestamp
def getTimestampDict(ds):
	result = dict()
	index = 0
	for tr in ds.tr_times:
		word_list = []
		for i in range(index, len(ds.data_times)):
			if ds.data_times[i] < tr:
				word_list.append(list(ds.data)[i])
				index += 1
			else:
				break
		result[tr] = word_list
	return result


#Concatenate a list of strings into one big string
def listToString(words):
	if len(words) <= 0:
		return ""
	result = ""
	for i in range(len(words) - 1):
		result += words[i] + " "
	result += words[len(words) - 1]
	return result

def numUniqueWords(sList):
	track = set()
	for string in sList:
		track.add(string)
	return len(track)