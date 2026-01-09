/* -*- mode: C++ -*- */
#include <getopt.h>
#include <inttypes.h>
#include <bitset>
#include <cmath>
#include <cassert>
#include "csr_graph.h"
extern int start_node;
extern char *INPUT, *OUTPUT;
extern int CUDA_DEVICE;
extern bool VERTEEX_REORDERING;
const char *prog_opts = "ls:d:";
const char *prog_usage = "[-s startNode]";

int QUIET = 0;
int process_prog_arg(int argc, char *argv[], int arg_start) {
	return 1;
}
void process_prog_opt(char c, char *optarg) {
	if (c == 's') {
		start_node = atoi(optarg);
		assert(start_node >= 0);
	}
}

void usage(int argc, char *argv[]) {
	fprintf(stderr, "usage: %s [-q] [-g gpunum] [-o output-file] [-v (do vertex reordering)] %s graph-file \n", argv[0], prog_usage);

}

void parse_args(int argc, char *argv[]) {
	int c;
	const char *skel_opts = "g:qo:v";
	char *opts;
	int len = 0;

	len = strlen(skel_opts) + strlen(prog_opts) + 1;
	opts = (char *) calloc(1, len);
	strcat(strcat(opts, skel_opts), prog_opts);

	while ((c = getopt(argc, argv, opts)) != -1) {
		switch (c) {
		case 'q':
			QUIET = 1;
			break;
		case 'o':
			OUTPUT = optarg; //TODO: copy?
			break;
		case 'v':
			VERTEEX_REORDERING = true;
			break;
		case 'g':
			char *end;
			errno = 0;
			CUDA_DEVICE = strtol(optarg, &end, 10);
			if (errno != 0 || *end != '\0') {
				fprintf(stderr, "Invalid GPU device '%s'. An integer must be specified.\n", optarg);
				exit (EXIT_FAILURE);
			}
			break;
		case '?':
			usage(argc, argv);
			exit (-1);
		default:
			process_prog_opt(c, optarg);
			break;
		}
	}

	if (optind < argc) {
		INPUT = argv[optind];
		if (!process_prog_arg(argc, argv, optind + 1)) {
			usage(argc, argv);
			exit (-1);
		}
	} else {
		usage(argc, argv);
		exit (-1);
	}
}

void output(CSRGraphTy &g, const char *output_file) {
	FILE *f;

	if (!output_file)
		return;

	if (strcmp(output_file, "-") == 0)
		f = stdout;
	else
		f = fopen(output_file, "w");

	for (int i = 0; i < g.nnodes; i++) {
		// If PRO is enabled, map original ID 'i' to its new location
		int current_id = i;
		if (g.old_to_new_mapping != NULL) {
			current_id = g.old_to_new_mapping[i];
		}

		if (g.node_data[current_id] == INF) {
			fprintf(f, "%d INF\n", i);
		} else {
			fprintf(f, "%d %d\n", i, g.node_data[current_id]);
		}
	}
	if (f != stdout) fclose(f);

}
