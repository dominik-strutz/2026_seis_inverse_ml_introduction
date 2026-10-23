"""try_it: the playground task list shows the question first, the recipe collapsed."""

from ml_seis.slides import try_it


def test_try_it_labels_and_body():
    html = try_it([("Capacity", "How small?", "Width 4.", "The dashed line.")]).text
    assert "Capacity." in html and "How small?" in html
    assert "Setup." in html and "Look for." in html
