#include <unity.h>
#include "mock_idf.h"

void setUp(void) {}
void tearDown(void) {}

void test_placeholder(void) { TEST_PASS(); }

void app_main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_placeholder);
    UNITY_END();
}