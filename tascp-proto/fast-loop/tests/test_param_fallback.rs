use fast_loop::params::Params;
// We need to expose modules as pub to test them directly, or test via HTTP.
// For now, this is a basic test. Real tests will be added.
#[test]
fn test_default_params() {
    let p = Params::default();
    assert_eq!(p.gamma, 0.9);
}
