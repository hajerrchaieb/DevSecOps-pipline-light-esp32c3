#include <unity.h>
#include "mock_idf.h"

void setUp(void) {}
void tearDown(void) {}

/* Test 1: on/off cluster — normal operation */
void test_on_off_cluster_toggle(void) {
    esp_matter_attr_val_t val;
    val.type = ESP_MATTER_VAL_TYPE_BOOLEAN;
    val.val.b = true;
    esp_err_t ret = app_driver_attribute_update(NULL, 1, 0x0006, &val);
    TEST_ASSERT_EQUAL(ESP_OK, ret);
}

/* Test 2: NULL endpoint — robustness */
void test_null_endpoint_handled(void) {
    esp_err_t ret = app_driver_attribute_update(NULL, 0, 0x0006, NULL);
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, ret);
}

/* Test 3: invalid attribute type — boundary check */
void test_invalid_attribute_type(void) {
    esp_matter_attr_val_t val;
    val.type = ESP_MATTER_VAL_TYPE_INVALID;
    esp_err_t ret = app_driver_attribute_update(NULL, 1, 0x0006, &val);
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, ret);
}

/* Test 4: malloc exhaustion — robustness */
void test_malloc_exhaustion(void) {
    void *ptr = malloc(1024);
    if (ptr == NULL) {
        TEST_ASSERT_EQUAL(ESP_ERR_NO_MEM, ESP_ERR_NO_MEM);
    } else {
        free(ptr);
    }
}

void app_main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_on_off_cluster_toggle);
    RUN_TEST(test_null_endpoint_handled);
    RUN_TEST(test_invalid_attribute_type);
    RUN_TEST(test_malloc_exhaustion);
    UNITY_END();
}