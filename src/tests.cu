/* Author: Garrett Scholtes
 * Date:   2015-12-02
 *
 * tests.cu - A test bed for the root-finding
 * functions, including benchmarks.
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <stdio.h>

#include "rootfinding.h"

int main(void) {

    printf("add2toThis(3) = %d\n", add2toThis(3));

    return 0;
}