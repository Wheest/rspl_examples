#include <libdragon.h>
#include <malloc.h>
#include <stdio.h>
#define ROUND_UP(x, s) (((x) + ((s)-1)) & ~((s)-1))

void do_something() {
  // Nested for loop to initialize an array with some random values
  // This is just to simulate some work being done
  // Setup randomness
  srand(0);
  int arr[1024];
  for (int i = 0; i < 1024; ++i) {
    for (int j = 0; j < 1024; ++j) {
      arr[j] = rand() % 256;
    }
  }
}

unsigned long read_serial(unsigned char *numbers, int num_values) {
  // Open the file for reading in binary mode
  FILE *fptr = fopen("rom:/integers.dat", "rb");
  if (!fptr) {
    printf("Error opening file\n");
  }

  size_t read_count;

  // Read the integers into the array
  unsigned long start = get_ticks();
  read_count = fread(numbers, sizeof(unsigned char), num_values, fptr);
  if (read_count != num_values) {
    printf("Error reading file\n");
    fclose(fptr);
  }

  // Close the file
  fclose(fptr);

  do_something();

  unsigned long time_spent_cpu = get_ticks() - start;
  printf("\nCPU baseline time (CPU ticks): %lu\n", time_spent_cpu);
  /* debug_hexdump(numbers, 1024); */
  return time_spent_cpu;
}

unsigned long read_async(unsigned char *numbers, int num_values) {
  // Read the integers into the array
  unsigned long start = get_ticks();

  uint32_t rom_add = dfs_rom_addr("integers.dat");
  // Need to change initial b with an 1
  // this might be patched out
  rom_add = rom_add & 0x1fffffff;

  dma_read_async(numbers, rom_add, num_values);

  do_something();
  dma_wait();
  unsigned long time_spent_cpu = get_ticks() - start;

  printf("\nAsync version time (CPU ticks): %lu\n", time_spent_cpu);
  return time_spent_cpu;
}

int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();
  dfs_init(DFS_DEFAULT_LOCATION);

  printf("Program starting\n");

  int num_values = 1024;
  unsigned char *numbers;
  numbers = (void *)memalign(16, ROUND_UP(num_values * sizeof(int), 16));

  unsigned long stime = read_serial(numbers, num_values);

  unsigned char *numbers2;
  // numbers2 = (void *)memalign(16, ROUND_UP(1024 * sizeof(int), 16));
  numbers2 = malloc_uncached_aligned(num_values * sizeof(unsigned char), 16);

  unsigned long atime = read_async(numbers2, num_values);

  // check that the data is the same
  for (int i = 0; i < num_values; ++i) {
    if (numbers[i] != numbers2[i]) {
      printf("Data mismatch at index %d\n", i);
      printf("Expected: %u, got: %u\n", numbers[i], numbers2[i]);
      printf("Peak-ahead values %u, %u\n", numbers[i + 1], numbers2[i + 1]);
      return EXIT_FAILURE;
    }
  }

  printf("\nDone and matching!\n");
  printf("Speedup: %f\n", (float)stime / atime);

  return EXIT_SUCCESS;
}
