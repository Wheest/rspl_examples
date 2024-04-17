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

void read_serial(unsigned char *numbers) {
  // Open the file for reading in binary mode
  FILE *fptr = fopen("rom:/integers.dat", "rb");
  if (!fptr) {
    printf("Error opening file\n");
  }

  size_t read_count;

  // Read the integers into the array
  unsigned long start = get_ticks();
  read_count = fread(numbers, sizeof(unsigned char), 1024, fptr);
  if (read_count != 1024) {
    printf("Error reading file\n");
    fclose(fptr);
  }

  // Close the file
  fclose(fptr);

  do_something();

  unsigned long time_spent_cpu = get_ticks() - start;
  printf("\nCPU baseline time (CPU ticks): %lu\n", time_spent_cpu);
}

void read_async(unsigned char *numbers) {
  // Read the integers into the array
  unsigned long start = get_ticks();

  uint32_t rom_add = dfs_rom_addr("rom:/integers.dat");

  dma_read_async(numbers, rom_add, 1024);

  do_something();
  dma_wait();
  unsigned long time_spent_cpu = get_ticks() - start;

  printf("\nAsync version time (CPU ticks): %lu\n", time_spent_cpu);
}

int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();
  dfs_init(DFS_DEFAULT_LOCATION);

  printf("Program starting\n");

  unsigned char *numbers;
  numbers = (void *)memalign(16, ROUND_UP(1024 * sizeof(int), 16));

  read_serial(numbers);

  unsigned char *numbers2;
  numbers2 = (void *)memalign(16, ROUND_UP(1024 * sizeof(int), 16));

  read_async(numbers2);

  for (int i = 0; i < 10; ++i) {
    printf("%u ", numbers2[i]);
  }

  // check that the data is the same
  for (int i = 0; i < 1024; ++i) {
    if (numbers[i] != numbers2[i]) {
      printf("Data mismatch at index %d\n", i);
      return EXIT_FAILURE;
    }
  }

  printf("Done\n");

  return EXIT_SUCCESS;
}
