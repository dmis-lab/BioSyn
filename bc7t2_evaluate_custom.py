import argparse
import codecs
import collections
import datetime
import json
import logging
import os
import random
import sys
import xml.etree.ElementTree as ElementTree

import lca

root = "MESH:ROOT"
log_format = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"

# Returns precision, recall & f-score for the specified reference and prediction files

log = logging.getLogger(__name__)

evaluation_config = collections.namedtuple("evaluation_config", ("annotation_type", "evaluation_type"))
evaluation_count = collections.namedtuple("evaluation_count", ("tp", "fp", "fn"))
evaluation_result = collections.namedtuple("evaluation_result", ("precision", "recall", "f_score"))
span_annotation = collections.namedtuple("span_annotation", ("document_id", "type", "locations", "text"))
identifier_annotation = collections.namedtuple("identifier_annotation", ("document_id", "type", "identifier"))
annotation_location = collections.namedtuple("annotation_location", ("offset", "length"))

def get_annotations_from_XML(input_collection, input_filename, eval_config):
	annotation_set = set()
	passage_text_dict = collections.defaultdict(dict)
	for document in input_collection.findall(".//document"):
		document_id = document.find(".//id").text
		for passage in document.findall(".//passage"):
			passage_offset = int(passage.find(".//offset").text)
			if passage.find(".//text") is None:
				continue
			passage_text = passage.find(".//text").text
			passage_text_dict[document_id][passage_offset] = passage_text
			for annotation in passage.findall(".//annotation"):
				type = annotation.find(".//infon[@key='type']").text
				if not eval_config.annotation_type is None and type != eval_config.annotation_type:
					continue
				if sum(int(location.get("length")) for location in annotation.findall(".//location")) == 0:
					continue
				if eval_config.evaluation_type == "span":
					locations = [annotation_location(int(location.get("offset")), int(location.get("length"))) for location in annotation.findall(".//location")]
					locations.sort()
					annotation_text = annotation.find(".//text").text
					location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
					annotation = span_annotation(document_id, type, tuple(locations), annotation_text)
					#log.debug("BioCXML file {} span annotation {}".format(input_filename, str(annotation)))
					if annotation_text != location_text:
						log.error("Annotation text {0} does not match text at location(s) {1} for annotation {2}".format(annotation_text, location_text, annotation))
					annotation_set.add(annotation)
				if eval_config.evaluation_type == "identifier":
					identifier_node = annotation.find(".//infon[@key='identifier']")
					if identifier_node is None:
						continue
					for identifier in identifier_node.text.split(","):
						annotation = identifier_annotation(document_id, type, identifier)
						#log.debug("BioCXML file {} identifier annotation {}".format(input_filename, str(annotation)))
						annotation_set.add(annotation)
	return annotation_set, passage_text_dict

def get_annotations_from_JSON(input_collection, input_filename, eval_config):
	annotation_set = set()
	passage_text_dict = collections.defaultdict(dict)
	for document in input_collection["documents"]:
		document_id = document["id"]
		for passage in document["passages"]:
			passage_offset = passage["offset"]
			passage_text = passage.get("text")
			# if passage_text is None:
			# 	continue
			# passage_text_dict[document_id][passage_offset] = passage_text
			for annotation in passage["annotations"]:
				type = annotation["infons"]["type"]
				if not eval_config.annotation_type is None and type != eval_config.annotation_type:
					continue
				# if sum(location["length"] for location in annotation["locations"]) == 0:
				# 	continue
				if eval_config.evaluation_type == "span":
					locations = [annotation_location(location["offset"], location["length"]) for location in annotation["locations"]]
					locations.sort()
					annotation_text = annotation["text"]
					location_text = " ".join([passage_text[offset - passage_offset: offset - passage_offset + length] for offset, length in locations])
					annotation = span_annotation(document["id"], type, tuple(locations), annotation_text)
					#log.debug("BioCJSON file {} span annotation {}".format(input_filename, str(annotation)))
					if annotation_text != location_text:
						log.error("Annotation text {0} does not match text at location(s) {1} for annotation {2}".format(annotation_text, location_text, annotation))
					annotation_set.add(annotation)
				if eval_config.evaluation_type == "identifier":
					for identifier in annotation["infons"]["identifier"].split(","):
						annotation = identifier_annotation(document["id"], type, identifier)
						#log.debug("BioCJSON file {} identifier annotation {}".format(input_filename, str(annotation)))
						annotation_set.add(annotation)
	return annotation_set, passage_text_dict
			
def get_annotations_from_file(input_filename, eval_config):
	try:
		if input_filename.endswith(".xml"):
			log.info("Reading XML file {}".format(input_filename))
			parser = ElementTree.XMLParser(encoding="utf-8")
			input_collection = ElementTree.parse(input_filename, parser=parser).getroot()
			return get_annotations_from_XML(input_collection, input_filename, eval_config)
		if input_filename.endswith(".json"):
			log.info("Reading JSON file {}".format(input_filename))
			with codecs.open(input_filename, 'r', encoding="utf8") as input:
				input_collection = json.load(input)
			return get_annotations_from_JSON(input_collection, input_filename, eval_config)
		log.info("Ignoring file {}".format(input_filename))
		return set(), dict()
	except Exception as e:
		raise RuntimeError("Error while processing file {}".format(input_filename)) from e

def get_annotations_from_path(input_path, eval_config):
	annotation_set = set()
	passage_text_dict = collections.defaultdict(set)
	if os.path.isdir(input_path):
		log.info("Processing directory {}".format(input_path))
		dir = os.listdir(input_path)
		for item in dir:
			input_filename = input_path + "/" + item
			if os.path.isfile(input_filename):
				annotation_set2, passage_text_dict2 = get_annotations_from_file(input_filename, eval_config)
				annotation_set.update(annotation_set2)
				passage_text_dict.update(passage_text_dict2)
	elif os.path.isfile(input_path):
		annotation_set2, passage_text_dict2 = get_annotations_from_file(input_path, eval_config)
		annotation_set.update(annotation_set2)
		passage_text_dict.update(passage_text_dict2)
	else:  
		raise RuntimeError("Path is not a directory or normal file: {}".format(input_path))
	return annotation_set, passage_text_dict

def calculate_evaluation_count(reference_annotations, predicted_annotations):
	reference_annotations = set(reference_annotations)
	predicted_annotations = set(predicted_annotations)
	annotations = set()
	annotations.update(reference_annotations)
	annotations.update(predicted_annotations)
	annotations = list(annotations)
	annotations.sort()
	results = collections.Counter()
	for a in annotations:
		r = a in reference_annotations
		p = a in predicted_annotations
		results[(r, p)] += 1
		log.debug("annotation = {} in reference = {} in predicted = {}".format(str(a), r, p))
	log.debug("Raw results = {}".format(str(results)))
	return evaluation_count(results[(True, True)], results[(False, True)], results[(True, False)])

def calculate_evaluation_result(eval_count):
	if eval_count.tp == 0:
		return evaluation_result(0.0, 0.0, 0.0)
	p = eval_count.tp / (eval_count.tp + eval_count.fp)
	r = eval_count.tp / (eval_count.tp + eval_count.fn)
	f = 2.0 * p * r / (p + r)
	return evaluation_result(p, r, f)

def do_strict_eval(reference_annotations, predicted_annotations):
	eval_count = calculate_evaluation_count(reference_annotations, predicted_annotations)
	log.info("TP = {0}, FP = {1}, FN = {2}".format(eval_count.tp, eval_count.fp, eval_count.fn))
	eval_result = calculate_evaluation_result(eval_count)
	return eval_result

def get_locations(annotations):
	locations = collections.defaultdict(set)
	for annotation in annotations:
		locations[annotation.document_id].update({(offset, offset + length) for offset, length in annotation.locations})
	return locations

def do_approx_span_eval(reference_annotations, predicted_annotations):
	tp1, fn = 0, 0
	predicted_locations = get_locations(predicted_annotations)
	for annotation in reference_annotations:
		predicted_locations2 = predicted_locations[annotation.document_id]
		found = False
		for location in annotation.locations:
			found |= any([location.offset < end2 and start2 < location.offset + location.length for start2, end2 in predicted_locations2])
		if found:
			tp1 += 1
		else:
			fn += 1
	log.info("REFERENCE: TP = {0}, FN = {1}".format(tp1, fn))
		
	tp2, fp = 0, 0
	reference_locations = get_locations(reference_annotations)
	for annotation in predicted_annotations:
		reference_locations2 = reference_locations[annotation.document_id]
		found = False
		for location in annotation.locations:
			found |= any([location.offset < end2 and start2 < location.offset + location.length for start2, end2 in reference_locations2])
		if found:
			tp2 += 1
		else:
			fp += 1
	log.info("PREDICTED: TP = {0}, FP = {1}".format(tp2, fp))

	if tp1 + tp2 == 0:
		return evaluation_result(0.0, 0.0, 0.0)
	p = tp2 / (tp2 + fp)
	r = tp1 / (tp1 + fn)
	f = 2.0 * p * r / (p + r)
	return evaluation_result(p, r, f)

def get_docid2identifiers(annotations):
	docid2identifiers = collections.defaultdict(set)
	for docid, type, identifier in annotations:
		if annotation_type == type:
			docid2identifiers[docid].add(identifier)
	return docid2identifiers

def do_approx_identifier_eval(lca_hierarchy, reference_annotations, predicted_annotations):
	reference_docid2identifiers = get_docid2identifiers(reference_annotations)
	predicted_docid2identifiers = get_docid2identifiers(predicted_annotations)
	docids = set(reference_docid2identifiers.keys())
	docids.update(predicted_docid2identifiers.keys())
	
	precision = list()
	recall = list()
	f_score = list()
	for docid in docids:
		log.info("Evaluating document {}".format(docid))
		reference_identifiers = reference_docid2identifiers[docid]
		predicted_identifiers = predicted_docid2identifiers[docid]
		reference_augmented, predicted_augmented = lca_hierarchy.get_augmented_sets(reference_identifiers, predicted_identifiers)
		log.info("{}: len(reference_identifiers) = {} len(reference_augmented) = {}".format(docid, len(reference_identifiers), len(reference_augmented)))
		log.info("{}: len(predicted_identifiers) = {} len(predicted_augmented) = {}".format(docid, len(predicted_identifiers), len(predicted_augmented)))
		eval_count = calculate_evaluation_count(reference_augmented, predicted_augmented)
		log.info("{}: TP = {}, FP = {}, FN = {}".format(docid, eval_count.tp, eval_count.fp, eval_count.fn))
		eval_result = calculate_evaluation_result(eval_count)
		log.info("{}: P = {:.4f}, R = {:.4f}, F = {:.4f}".format(docid, eval_result.precision, eval_result.recall, eval_result.f_score))
		precision.append(eval_result.precision)
		recall.append(eval_result.recall)
		f_score.append(eval_result.f_score)
	avg_precision = sum(precision) / len(precision)
	avg_recall = sum(recall) / len(recall)
	avg_f_score = sum(f_score) / len(f_score)
	return evaluation_result(avg_precision, avg_recall, avg_f_score)

def verify_document_sets(reference_passages, predicted_passages):
	verification_errors = list()
	# Verify that reference path and prediction path contain the same set of documents
	reference_docids = set(reference_passages.keys())
	predicted_docids = set(predicted_passages.keys())
	if len(reference_docids - predicted_docids) > 0:
		verification_errors.append("Prediction path is missing documents {}".format(", ".join(reference_docids - predicted_docids)))
	if len(predicted_docids - reference_docids) > 0:
		verification_errors.append("Prediction path contains extra documents {}".format(", ".join(predicted_docids - reference_docids)))
	# Verify that the reference and predicted files are the same
	docids = reference_docids.intersection(predicted_docids)
	for document_id in docids:
		reference_passage_offsets = set(reference_passages[document_id].keys())
		predicted_passage_offsets = set(predicted_passages[document_id].keys())
		if len(reference_passage_offsets) != len(predicted_passage_offsets):
			verification_errors.append("Number of passages does not match for document {0}, {1} != {2}".format(document_id, len(reference_passage_offsets), len(predicted_passage_offsets)))
		elif reference_passage_offsets != predicted_passage_offsets:
			verification_errors.append("Passage offsets do not match for document {}".format(document_id))
		else:
			for offset in reference_passage_offsets:
				if reference_passages[document_id][offset] != predicted_passages[document_id][offset]:
					verification_errors.append("Passage text does not match for document {0}, offset {1}".format(document_id, offset))
	return verification_errors

def bc7t2_evaluate(reference_path, prediction_path):
	evaluation_type = 'identifier'
	annotation_type = 'Chemical'
	eval_config = evaluation_config(annotation_type, evaluation_type)
	reference_annotations, reference_passages = get_annotations_from_path(reference_path, eval_config)
	predicted_annotations, predicted_passages = get_annotations_from_path(prediction_path, eval_config)
	
	eval_result = do_strict_eval(reference_annotations, predicted_annotations)
	print("P = {0:.4f}, R = {1:.4f}, F = {2:.4f}".format(eval_result.precision, eval_result.recall, eval_result.f_score))

	return eval_result.precision, eval_result.recall, eval_result.f_score
	
# if __name__ == "__main__":
# 	start = datetime.datetime.now()
# 	parser = argparse.ArgumentParser(description="Evaluation script for BioCreative 7 Track 2: NLM Chem task")
# 	parser.add_argument("--reference_path", "-r", type=str, required=True, help="path to directory or file containing the reference annotations, i.e. the annotations considered correct")
# 	parser.add_argument("--prediction_path", "-p", type=str, required=True, help="path to directory or file containing the predicted annotations, i.e. the annotations being evaluated")
# 	parser.add_argument("--parents_filename", "-f", type=str, default=None, help="name of file containing MeSH IDs and their parents, only used for approximate identifier evaluation")
# 	parser.add_argument("--evaluation_type", "-t", choices = {"span", "identifier"}, required=True, help="The type of evaluation to perform")
# 	parser.add_argument("--evaluation_method", "-m", choices = {"strict", "approx"}, required=True, help="Whether to perform a strict or approximate evaluation")
# 	parser.add_argument("--annotation_type", "-a", type=str, required=True, help="The annotation type to consider, all others are ignored. 'None' considers all types, but it still must match")
# 	parser.add_argument("--logging_level", "-l", type=str, default="INFO", help="The logging level, options are {critical, error, warning, info, debug}")
# 	parser.add_argument("--no_document_verification", dest='verify_documents', action='store_const', const=False, default=True, help='Do not verify that reference and predicted document sets match')
	
# 	args = parser.parse_args()
# 	evaluation_type = args.evaluation_type
# 	evaluation_method = args.evaluation_method
# 	annotation_type = args.annotation_type if not args.annotation_type.lower() == "none" else None
# 	logging.basicConfig(level=args.logging_level.upper(), format=log_format)
	
# 	if log.isEnabledFor(logging.DEBUG):
# 		for arg, value in sorted(vars(args).items()):
# 			log.info("Argument {0}: {1}".format(arg, value))

# 	eval_config = evaluation_config(annotation_type, evaluation_type)
# 	reference_annotations, reference_passages = get_annotations_from_path(args.reference_path, eval_config)
# 	predicted_annotations, predicted_passages = get_annotations_from_path(args.prediction_path, eval_config)
# 	if args.verify_documents:
# 		verification_errors = verify_document_sets(reference_passages, predicted_passages)
# 		for verification_error in verification_errors:
# 			log.error(verification_error)
# 		if len(verification_errors) > 0:
# 			sys.exit(1)

# 	if evaluation_method == "strict":
# 		eval_result = do_strict_eval(reference_annotations, predicted_annotations)
# 	elif evaluation_method == "approx" and evaluation_type == "span":
# 		eval_result = do_approx_span_eval(reference_annotations, predicted_annotations)
# 	elif evaluation_method == "approx" and evaluation_type == "identifier":
# 		if args.parents_filename is None:
# 			raise RuntimeError("Approximate identifier evaluation requires a parents filename")
# 		lca_hierarchy = lca.lca_hierarchy(root)
# 		lca_hierarchy.load_parents(args.parents_filename)
# 		eval_result = do_approx_identifier_eval(lca_hierarchy, reference_annotations, predicted_annotations)
# 	else:
# 		raise ValueError("Unknown evaluation method: {}".format(evaluation_method))
# 	print("P = {0:.4f}, R = {1:.4f}, F = {2:.4f}".format(eval_result.precision, eval_result.recall, eval_result.f_score))
# 	print("Elapsed time: {}".format(datetime.datetime.now() - start))